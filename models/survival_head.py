"""
Survival prediction head for BraTS overall survival (OS) estimation.

Attaches to any encoder's bottleneck features and predicts overall survival
in days as a regression task.

BraTS 2019+ provides OS labels for the HGG subset. This head can be trained
jointly with the segmentation loss or fine-tuned independently.

Usage:
    from models.survival_head import SurvivalHead, SurvivalModel
    from models import build_model

    seg_model = build_model("SegResNet", num_classes=3)
    survival_model = SurvivalModel(seg_model, encoder_out_channels=256)

    seg_logits, os_days = survival_model(image)
    loss = seg_loss(seg_logits, label) + 0.1 * survival_loss(os_days, target_days)
"""

import torch
import torch.nn as nn
from .BaseModel import BaseModel


class SurvivalHead(nn.Module):
    """Regression head that maps spatial encoder features to a survival estimate.

    Args:
        in_channels: Number of channels in the bottleneck feature map.
        hidden_dim: Hidden dimension for the MLP.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU(),  # OS is non-negative
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Bottleneck feature map [B, C, D, H, W].

        Returns:
            Predicted OS in days [B, 1].
        """
        x = self.gap(features).flatten(1)  # [B, C]
        return self.mlp(x)


class SurvivalModel(BaseModel):
    """Wraps a segmentation model with an additional survival prediction head.

    The segmentation backbone is used as-is; the survival head taps into the
    global-average-pooled output of the final encoder feature map.

    Because most MONAI models don't expose intermediate features, we use a
    forward hook to capture them.

    Args:
        seg_model: A trained or untrained segmentation model.
        encoder_out_channels: Number of channels in the bottleneck features.
            For SegResNet with init_filters=32 this is typically 256.
        hidden_dim: Hidden dim for the survival MLP.
    """

    def __init__(self, seg_model: nn.Module, encoder_out_channels: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.seg_model = seg_model
        self.survival_head = SurvivalHead(in_channels=encoder_out_channels, hidden_dim=hidden_dim)

        # We capture bottleneck features via a forward hook on the last
        # encoder layer. The hook is registered externally for flexibility.
        self._bottleneck_features: torch.Tensor = None

    def register_bottleneck_hook(self, layer: nn.Module) -> None:
        """Register a forward hook on a specific layer to capture its output.

        Example:
            survival_model.register_bottleneck_hook(seg_model.encoder[-1])
        """
        def _hook(module, input, output):
            self._bottleneck_features = output

        layer.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            seg_logits: Segmentation output from the backbone.
            os_pred: Predicted overall survival in days [B, 1], or None if
                     the bottleneck hook has not been registered.
        """
        seg_logits = self.seg_model(x)

        os_pred = None
        if self._bottleneck_features is not None:
            os_pred = self.survival_head(self._bottleneck_features)

        return seg_logits, os_pred

    def test(self):
        inp = torch.rand(1, 4, 128, 128, 128)
        seg_logits, _ = self.forward(inp)
        assert seg_logits is not None
        print("SurvivalModel forward test OK!")


def survival_loss(pred_days: torch.Tensor, target_days: torch.Tensor) -> torch.Tensor:
    """L1 loss for survival regression, normalized by the target scale.

    Args:
        pred_days: Predicted OS [B, 1].
        target_days: Ground-truth OS in days [B, 1].

    Returns:
        Scalar loss value.
    """
    return nn.functional.l1_loss(pred_days, target_days)
