"""
BraTS challenge evaluation metrics.

The official BraTS evaluation uses:
  - Dice Similarity Coefficient (DSC) per class
  - Hausdorff Distance 95th percentile (HD95) per class
  - Sensitivity (recall)
  - Specificity

Computed for each of: TC (Tumor Core), WT (Whole Tumor), ET (Enhancing Tumor)

Usage:
    from utils.metrics import BraTSMetrics

    metrics = BraTSMetrics()
    results = metrics.compute(pred_binary, target_binary)
    # results = {"TC_dice": 0.89, "TC_hd95": 3.2, "WT_dice": 0.91, ...}
"""

import torch
import numpy as np
from typing import Dict, Optional
from scipy.ndimage import distance_transform_edt


CLASS_NAMES = ["TC", "WT", "ET"]


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient for binary masks.

    Returns 1.0 if both masks are empty (no tumor in GT and no false positive).
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    intersection = (pred & target).sum()
    return 2.0 * intersection / (pred.sum() + target.sum())


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute the 95th percentile Hausdorff Distance.

    Returns 0.0 if both masks are empty.
    Returns a large value (373.13 — BraTS convention for 240x240x155 volumes)
    if one mask is empty and the other is not.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    if pred.sum() == 0 or target.sum() == 0:
        return 373.13  # BraTS convention: max possible HD for 240x240x155

    # Surface voxels: boundary of each mask
    pred_border = _get_border(pred)
    target_border = _get_border(target)

    # Distance transforms
    dt_pred = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)

    # Distances from surface of pred to nearest surface of target (and vice versa)
    d_pred_to_target = dt_target[pred_border]
    d_target_to_pred = dt_pred[target_border]

    if len(d_pred_to_target) == 0 or len(d_target_to_pred) == 0:
        return 373.13

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_distances, 95))


def sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """True positive rate: TP / (TP + FN)."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    if target.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0

    tp = (pred & target).sum()
    return float(tp / target.sum())


def specificity(pred: np.ndarray, target: np.ndarray) -> float:
    """True negative rate: TN / (TN + FP)."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    neg = (~target).sum()
    if neg == 0:
        return 1.0

    tn = (~pred & ~target).sum()
    return float(tn / neg)


def _get_border(mask: np.ndarray) -> np.ndarray:
    """Extract surface (border) voxels from a binary 3D mask.

    A voxel is on the border if it's True and has at least one False neighbor.
    """
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(mask, iterations=1)
    border = mask & ~eroded
    return border


class BraTSMetrics:
    """Compute full BraTS evaluation metrics for a batch of predictions.

    Expects binary masks with shape [B, 3, D, H, W] where channel order
    is [TC, WT, ET].
    """

    def __init__(self, class_names: Optional[list] = None):
        self.class_names = class_names or CLASS_NAMES

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute metrics for a single sample.

        Args:
            pred: Binary prediction [C, D, H, W] or [1, C, D, H, W].
            target: Binary ground truth [C, D, H, W] or [1, C, D, H, W].

        Returns:
            Dict with keys like "TC_dice", "TC_hd95", "TC_sensitivity", etc.
        """
        if pred.ndim == 5:
            pred = pred[0]
        if target.ndim == 5:
            target = target[0]

        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        results = {}
        for i, name in enumerate(self.class_names):
            p = pred_np[i]
            t = target_np[i]
            results[f"{name}_dice"] = dice_score(p, t)
            results[f"{name}_hd95"] = hausdorff_distance_95(p, t)
            results[f"{name}_sensitivity"] = sensitivity(p, t)
            results[f"{name}_specificity"] = specificity(p, t)

        # Mean Dice across all classes (primary BraTS ranking metric)
        results["mean_dice"] = np.mean(
            [results[f"{name}_dice"] for name in self.class_names]
        )
        results["mean_hd95"] = np.mean(
            [results[f"{name}_hd95"] for name in self.class_names]
        )
        return results

    def compute_batch(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute average metrics over a batch [B, C, D, H, W].

        Returns:
            Dict with averaged metric values.
        """
        batch_results = []
        for i in range(preds.shape[0]):
            batch_results.append(self.compute(preds[i], targets[i]))

        avg = {}
        for key in batch_results[0]:
            avg[key] = float(np.mean([r[key] for r in batch_results]))
        return avg


def compute_distance_map(target: np.ndarray) -> np.ndarray:
    """Compute signed distance transform for BoundaryLoss / HausdorffDTLoss.

    Args:
        target: Binary ground truth [C, D, H, W].

    Returns:
        Signed distance map [C, D, H, W]. Positive inside, negative outside.
    """
    dist_map = np.zeros_like(target, dtype=np.float32)
    for c in range(target.shape[0]):
        if target[c].sum() > 0:
            pos_dist = distance_transform_edt(target[c])
            neg_dist = distance_transform_edt(1 - target[c])
            dist_map[c] = pos_dist - neg_dist
        # If empty, dist_map stays all zeros
    return dist_map
