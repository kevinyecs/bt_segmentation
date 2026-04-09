"""
Inference utilities: sliding window and model ensembling.

Sliding window inference allows processing full-resolution MRI volumes
without OOM errors by predicting overlapping patches and stitching them.

Model ensembling averages predictions from multiple architectures for
improved accuracy (used by all BraTS winning solutions).

Usage:
    from utils.inference import sliding_window_predict, ensemble_predict

    # Single model, full-resolution sliding window
    pred = sliding_window_predict(model, full_res_image, roi_size=(128,128,128))

    # Multi-model ensemble
    models = [segresnet, swinunetr, mednext]
    pred = ensemble_predict(models, image)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch.cuda.amp import autocast


def sliding_window_predict(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    use_amp: bool = True,
    mode: str = "gaussian",
    device: Optional[str] = None,
) -> torch.Tensor:
    """Run sliding window inference using MONAI's optimized implementation.

    This is essential for full-resolution BraTS volumes (240x240x155) which
    don't fit in GPU memory at once when using roi_size=(128,128,128).

    Args:
        model: Segmentation model.
        image: Input volume [B, C, D, H, W].
        roi_size: Patch size for each prediction window.
        overlap: Fraction of overlap between adjacent patches (0.5 = 50%).
        use_amp: Use automatic mixed precision.
        mode: Blending mode — "gaussian" (weighted) or "constant".
        device: Target device. Defaults to model device.

    Returns:
        Predictions [B, classes, D, H, W] on CPU.
    """
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as e:
        raise ImportError("MONAI is required for sliding window inference") from e

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    image = image.to(device)

    def _forward(x):
        if use_amp and device != "cpu" and str(device) != "cpu":
            with autocast():
                out = model(x)
        else:
            out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=_forward,
            overlap=overlap,
            mode=mode,
        )

    return torch.sigmoid(pred).cpu()


def ensemble_predict(
    models: List[nn.Module],
    image: torch.Tensor,
    weights: Optional[List[float]] = None,
    use_sliding_window: bool = True,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    use_amp: bool = True,
) -> torch.Tensor:
    """Average predictions from multiple models (optionally weighted).

    BraTS winners typically ensemble 3-5 models with different architectures
    and/or folds for consistent accuracy gains (+1-2% Dice).

    Args:
        models: List of segmentation models.
        image: Input volume [B, C, D, H, W].
        weights: Per-model weights (None = equal weighting).
        use_sliding_window: Use sliding window for each model.
        roi_size: ROI size for sliding window.
        overlap: Overlap fraction.
        use_amp: Use automatic mixed precision.

    Returns:
        Averaged predictions [B, classes, D, H, W] on CPU.
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    preds = []
    for model, weight in zip(models, weights):
        if use_sliding_window:
            pred = sliding_window_predict(
                model, image,
                roi_size=roi_size,
                overlap=overlap,
                use_amp=use_amp,
            )
        else:
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                if use_amp and str(device) != "cpu":
                    with autocast():
                        pred = model(image.to(device))
                else:
                    pred = model(image.to(device))
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = torch.sigmoid(pred).cpu()

        preds.append(pred * weight)

    return sum(preds)


def ensemble_predict_with_tta(
    models: List[nn.Module],
    image: torch.Tensor,
    weights: Optional[List[float]] = None,
    use_amp: bool = True,
) -> torch.Tensor:
    """Ensemble + TTA: run each model with test-time augmentation, then average.

    This is the full inference pipeline used by top BraTS solutions:
    N models x 8 TTA variants = 8N predictions averaged.

    Memory-intensive but gives the best possible accuracy.

    Args:
        models: List of segmentation models.
        image: Input volume [B, C, D, H, W].
        weights: Per-model weights.
        use_amp: Use automatic mixed precision.

    Returns:
        Averaged predictions [B, classes, D, H, W] on CPU.
    """
    from utils.tta import tta_predict

    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    preds = []
    for model, weight in zip(models, weights):
        pred = tta_predict(model, image, use_amp=use_amp)
        preds.append(pred * weight)

    return sum(preds)
