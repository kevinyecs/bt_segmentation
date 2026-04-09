"""
Test-Time Augmentation (TTA) for 3D segmentation.

Averages predictions over a set of deterministic flips/rotations to improve
robustness without retraining.

Usage:
    from utils.tta import tta_predict

    pred = tta_predict(model, input_tensor)
"""

import torch
import torch.nn as nn
from typing import List, Callable


# Each augmentation is (forward_fn, inverse_fn)
# forward_fn: input → augmented input
# inverse_fn: augmented prediction → original-space prediction
_FLIP_AXES = [
    # flip axis 2 (D)
    (
        lambda x: torch.flip(x, [2]),
        lambda p: torch.flip(p, [2]),
    ),
    # flip axis 3 (H)
    (
        lambda x: torch.flip(x, [3]),
        lambda p: torch.flip(p, [3]),
    ),
    # flip axis 4 (W)
    (
        lambda x: torch.flip(x, [4]),
        lambda p: torch.flip(p, [4]),
    ),
    # flip D+H
    (
        lambda x: torch.flip(x, [2, 3]),
        lambda p: torch.flip(p, [2, 3]),
    ),
    # flip D+W
    (
        lambda x: torch.flip(x, [2, 4]),
        lambda p: torch.flip(p, [2, 4]),
    ),
    # flip H+W
    (
        lambda x: torch.flip(x, [3, 4]),
        lambda p: torch.flip(p, [3, 4]),
    ),
    # flip all three
    (
        lambda x: torch.flip(x, [2, 3, 4]),
        lambda p: torch.flip(p, [2, 3, 4]),
    ),
]


def tta_predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    use_amp: bool = True,
    include_original: bool = True,
) -> torch.Tensor:
    """Average predictions over mirrored versions of the input.

    Args:
        model: Segmentation model. Should be in eval mode.
        input_tensor: Input volume [B, C, D, H, W].
        use_amp: Use automatic mixed precision on CUDA.
        include_original: Include the un-augmented prediction in the average.

    Returns:
        Averaged sigmoid prediction [B, classes, D, H, W].
    """
    model.eval()
    device = next(model.parameters()).device
    x = input_tensor.to(device)

    augmentations = list(_FLIP_AXES)

    preds: List[torch.Tensor] = []

    with torch.no_grad():
        if include_original:
            preds.append(_forward(model, x, use_amp))

        for aug_fn, inv_fn in augmentations:
            x_aug = aug_fn(x)
            pred_aug = _forward(model, x_aug, use_amp)
            preds.append(inv_fn(pred_aug))

    stacked = torch.stack(preds, dim=0)  # [N, B, C, D, H, W]
    return stacked.mean(dim=0).cpu()


def _forward(model: nn.Module, x: torch.Tensor, use_amp: bool) -> torch.Tensor:
    device = next(model.parameters()).device
    if use_amp and device.type == "cuda":
        from torch.cuda.amp import autocast
        with autocast():
            out = model(x)
    else:
        out = model(x)

    if isinstance(out, tuple):
        out = out[0]

    return torch.sigmoid(out)
