"""
Monte Carlo Dropout for predictive uncertainty estimation.

Usage:
    from utils.uncertainty import mc_dropout_predict, uncertainty_map

    mean, variance = mc_dropout_predict(model, input_tensor, n_samples=20)
    unc = uncertainty_map(variance)  # per-voxel uncertainty
"""

import torch
import torch.nn as nn
from typing import Tuple


def _enable_dropout(model: nn.Module) -> None:
    """Set Dropout layers to train mode so they remain active during inference."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_dropout_predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_samples: int = 20,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run N stochastic forward passes to estimate prediction mean and variance.

    The model is kept in eval mode (BatchNorm uses running stats) but Dropout
    layers are re-enabled so each pass produces a different sample.

    Args:
        model: Any segmentation model (should contain at least one Dropout layer
               for uncertainty to be meaningful; works even without one).
        input_tensor: Input volume [B, C, H, W, D].
        n_samples: Number of stochastic forward passes.
        use_amp: Use automatic mixed precision on CUDA.

    Returns:
        mean: Averaged prediction [B, classes, H, W, D].
        variance: Per-voxel variance across samples [B, classes, H, W, D].
    """
    model.eval()
    _enable_dropout(model)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            if use_amp and device.type == "cuda":
                from torch.cuda.amp import autocast
                with autocast():
                    out = model(input_tensor)
            else:
                out = model(input_tensor)

            if isinstance(out, tuple):
                out = out[0]

            samples.append(torch.sigmoid(out).cpu())

    stacked = torch.stack(samples, dim=0)  # [N, B, C, H, W, D]
    mean = stacked.mean(dim=0)
    variance = stacked.var(dim=0)
    return mean, variance


def uncertainty_map(variance: torch.Tensor) -> torch.Tensor:
    """Collapse per-class variance into a single voxel-level uncertainty score.

    Uses mean variance across all output channels.

    Args:
        variance: [B, C, H, W, D] variance tensor from mc_dropout_predict.

    Returns:
        [B, H, W, D] uncertainty map in [0, 1].
    """
    unc = variance.mean(dim=1)  # average over classes → [B, H, W, D]
    # Normalize to [0, 1] per sample
    b = unc.shape[0]
    unc_flat = unc.view(b, -1)
    unc_min = unc_flat.min(dim=1).values.view(b, 1, 1, 1)
    unc_max = unc_flat.max(dim=1).values.view(b, 1, 1, 1)
    return (unc - unc_min) / (unc_max - unc_min + 1e-8)
