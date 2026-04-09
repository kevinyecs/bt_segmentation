"""
Post-processing utilities for brain tumor segmentation predictions.

Winning BraTS solutions apply post-processing to clean up predictions:
  - Remove small connected components (spurious false positives)
  - Enforce anatomical constraints (ET must be inside TC, TC inside WT)
  - Fill holes in predictions
  - Threshold probability maps

Usage:
    from utils.postprocess import postprocess_brats

    raw_pred = model.inference(image)                   # [B, 3, D, H, W] sigmoid
    clean_pred = postprocess_brats(raw_pred, threshold=0.5, min_size=100)
"""

import numpy as np
import torch
from typing import Optional
from scipy.ndimage import label as scipy_label, binary_fill_holes


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """Remove connected components smaller than min_size voxels.

    Args:
        mask: Binary 3D mask [D, H, W].
        min_size: Minimum component size in voxels.

    Returns:
        Cleaned binary mask.
    """
    labeled, num_features = scipy_label(mask)
    if num_features == 0:
        return mask

    cleaned = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() >= min_size:
            cleaned[component] = 1

    return cleaned


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component.

    Useful for whole tumor (WT) where there should be a single contiguous region.
    """
    labeled, num_features = scipy_label(mask)
    if num_features <= 1:
        return mask

    largest = 0
    largest_size = 0
    for i in range(1, num_features + 1):
        size = (labeled == i).sum()
        if size > largest_size:
            largest_size = size
            largest = i

    return (labeled == largest).astype(mask.dtype)


def fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a 3D binary mask (slice-wise for speed)."""
    filled = np.copy(mask)
    for z in range(mask.shape[0]):
        filled[z] = binary_fill_holes(mask[z])
    return filled


def enforce_brats_hierarchy(
    tc: np.ndarray,
    wt: np.ndarray,
    et: np.ndarray,
) -> tuple:
    """Enforce anatomical constraints between BraTS sub-regions.

    Constraints:
      - ET (enhancing tumor) must be a subset of TC (tumor core)
      - TC must be a subset of WT (whole tumor)

    Args:
        tc, wt, et: Binary 3D masks [D, H, W].

    Returns:
        Corrected (tc, wt, et) tuple.
    """
    # ET must be inside TC
    et = et & tc

    # TC must be inside WT
    tc = tc & wt

    # ET must still be inside (corrected) TC
    et = et & tc

    return tc, wt, et


def postprocess_brats(
    pred: torch.Tensor,
    threshold: float = 0.5,
    min_size_tc: int = 100,
    min_size_wt: int = 200,
    min_size_et: int = 50,
    fill_holes: bool = True,
    enforce_hierarchy: bool = True,
) -> torch.Tensor:
    """Full BraTS post-processing pipeline.

    Args:
        pred: Sigmoid predictions [B, 3, D, H, W] or [3, D, H, W].
              Channel order: [TC, WT, ET].
        threshold: Binarization threshold.
        min_size_tc: Min voxels for TC components.
        min_size_wt: Min voxels for WT components.
        min_size_et: Min voxels for ET components.
        fill_holes: Apply hole filling.
        enforce_hierarchy: Enforce ET ⊂ TC ⊂ WT.

    Returns:
        Binary predictions with same shape as input.
    """
    squeeze = False
    if pred.ndim == 4:
        pred = pred.unsqueeze(0)
        squeeze = True

    pred_np = pred.cpu().numpy()
    result = np.zeros_like(pred_np)

    for b in range(pred_np.shape[0]):
        tc = (pred_np[b, 0] >= threshold).astype(np.uint8)
        wt = (pred_np[b, 1] >= threshold).astype(np.uint8)
        et = (pred_np[b, 2] >= threshold).astype(np.uint8)

        # Remove small spurious components
        tc = remove_small_components(tc, min_size=min_size_tc)
        wt = remove_small_components(wt, min_size=min_size_wt)
        et = remove_small_components(et, min_size=min_size_et)

        # Fill holes
        if fill_holes:
            wt = fill_holes_3d(wt)
            tc = fill_holes_3d(tc)

        # Enforce anatomical hierarchy
        if enforce_hierarchy:
            tc, wt, et = enforce_brats_hierarchy(tc, wt, et)

        result[b, 0] = tc
        result[b, 1] = wt
        result[b, 2] = et

    output = torch.from_numpy(result).float()
    if squeeze:
        output = output.squeeze(0)
    return output
