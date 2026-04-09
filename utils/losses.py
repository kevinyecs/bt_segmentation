"""
Advanced loss functions for brain tumor segmentation.

BraTS winning solutions typically use compound losses that combine
region-based (Dice) with distribution-based (CE/Focal) and optionally
boundary-aware terms.

Usage:
    from utils.losses import DiceFocalLoss, DiceCELoss, BoundaryLoss, CompoundBraTSLoss

    criterion = CompoundBraTSLoss(dice_weight=1.0, focal_weight=1.0, boundary_weight=0.5)
    loss = criterion(logits, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Works with sigmoid outputs (multi-label) where each channel is
    an independent binary segmentation (TC, WT, ET for BraTS).

    Args:
        smooth: Smoothing factor to avoid division by zero.
        square: If True, use squared denominator (Dice-Sorensen vs Dice).
    """

    def __init__(self, smooth: float = 1.0, square: bool = False):
        super().__init__()
        self.smooth = smooth
        self.square = square

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.ndim))  # spatial dims only

        intersection = (probs * targets).sum(dim=dims)
        if self.square:
            cardinality = (probs.pow(2) + targets.pow(2)).sum(dim=dims)
        else:
            cardinality = (probs + targets).sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in segmentation.

    Down-weights well-classified voxels to focus on hard examples.
    Especially useful for small tumor sub-regions (ET).

    Args:
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter (higher = more focus on hard examples).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        return (focal_weight * ce).mean()


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss — used by many BraTS 2023-2024 winners.

    Args:
        dice_weight: Weight for Dice loss component.
        focal_weight: Weight for Focal loss component.
        focal_gamma: Focal loss gamma parameter.
    """

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0, focal_gamma: float = 2.0):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)


class DiceCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss.

    Simpler alternative to DiceFocalLoss, often used as a baseline.
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * ce


class BoundaryLoss(nn.Module):
    """Boundary (surface) loss that penalizes predictions far from the true boundary.

    Computes the distance transform of the ground truth and uses it to weight
    the prediction — voxels far from the boundary get penalized more.

    This loss is additive with Dice/CE and should be weighted small (0.1-0.5)
    and optionally ramped up during training.

    Reference: Kervadec et al., "Boundary loss for highly unbalanced segmentation"
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output [B, C, D, H, W].
            dist_maps: Signed distance transform of the ground truth [B, C, D, H, W].
                       Positive inside the object, negative outside.
                       Precompute using scipy.ndimage.distance_transform_edt.
        """
        probs = torch.sigmoid(logits)
        return (probs * dist_maps).mean()


class HausdorffDTLoss(nn.Module):
    """Hausdorff Distance loss via distance transforms.

    Approximates the Hausdorff distance using distance transforms of the
    prediction and ground truth, making it differentiable.

    Reference: Karimi & Salcudean, "Reducing the Hausdorff Distance in
    Medical Image Segmentation with Convolutional Neural Networks"
    """

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_dist_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw model output [B, C, D, H, W].
            targets: Binary ground truth [B, C, D, H, W].
            target_dist_maps: Distance transform of target [B, C, D, H, W].
        """
        probs = torch.sigmoid(logits)

        # Approximate distance transform of prediction using erosion-based approx
        # For true differentiable HD, we use the GT distance map weighted by pred error
        pred_error = (probs - targets).pow(2)
        weighted_error = pred_error * target_dist_maps.pow(self.alpha)
        return weighted_error.mean()


class CompoundBraTSLoss(nn.Module):
    """Full compound loss used by competitive BraTS solutions.

    Combines:
      - Dice loss (region overlap)
      - Focal loss (hard example mining)
      - Boundary loss (surface accuracy, optional)

    The boundary loss requires pre-computed distance maps and should be
    ramped up during training (start with 0, increase to boundary_weight
    over the first ~50% of training).

    Args:
        dice_weight: Weight for Dice component.
        focal_weight: Weight for Focal component.
        boundary_weight: Weight for Boundary component.
        focal_gamma: Focal loss gamma.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 0.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.boundary = BoundaryLoss() if boundary_weight > 0 else None
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dist_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.dice_weight * self.dice(logits, targets)
        loss = loss + self.focal_weight * self.focal(logits, targets)

        if self.boundary is not None and dist_maps is not None:
            loss = loss + self.boundary_weight * self.boundary(logits, dist_maps)

        return loss
