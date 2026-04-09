"""
Deep supervision wrapper for encoder-decoder segmentation models.

Deep supervision adds auxiliary segmentation heads at intermediate decoder
resolutions. Each head produces a prediction that is upsampled to the full
resolution and compared against the ground truth with exponentially decaying
loss weights (1.0, 0.5, 0.25, 0.125, ...).

This forces intermediate features to be semantically meaningful, improving
convergence speed and final accuracy (+1-2% Dice on BraTS).

Used by: nnU-Net, SwinUNETR (MONAI), most BraTS 2023-2024 winning solutions.

Usage:
    from models.deep_supervision import DeepSupervisionWrapper, deep_supervision_loss

    # Wrap any model that returns multi-scale features
    model = DeepSupervisionWrapper(base_model, feature_channels=[256, 128, 64, 32], num_classes=3)
    outputs = model(x)  # list of [B, C, D, H, W] at decreasing resolutions

    loss = deep_supervision_loss(outputs, target, criterion, weights=[1, 0.5, 0.25, 0.125])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DeepSupervisionHead(nn.Module):
    """Auxiliary segmentation head: 1x1 conv -> upsample to target size."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, ...]) -> torch.Tensor:
        x = self.conv(x)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return x


class DeepSupervisionWrapper(nn.Module):
    """Wraps any segmentation model to add deep supervision outputs.

    The base model must expose intermediate decoder features. This wrapper
    adds 1x1 conv heads at each intermediate resolution and returns a list
    of predictions (full-res first, then auxiliary at decreasing resolution).

    For models that don't expose intermediates natively, use forward hooks
    (see `register_feature_hooks()`).

    Args:
        base_model: The segmentation backbone.
        feature_channels: Channel counts at each decoder level (deepest first).
            E.g., for a model with decoder levels at 256, 128, 64, 32 channels.
        num_classes: Number of segmentation output channels.
        num_aux_heads: Number of auxiliary heads to add. If None, adds one per
            feature level.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_channels: List[int],
        num_classes: int = 3,
        num_aux_heads: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        n_heads = num_aux_heads or len(feature_channels)
        n_heads = min(n_heads, len(feature_channels))

        self.aux_heads = nn.ModuleList([
            DeepSupervisionHead(ch, num_classes)
            for ch in feature_channels[:n_heads]
        ])

        # Storage for intermediate features captured by hooks
        self._intermediate_features: List[torch.Tensor] = []

    def register_feature_hooks(self, layers: List[nn.Module]) -> None:
        """Register forward hooks on decoder layers to capture intermediate features.

        Args:
            layers: List of decoder sub-modules (one per resolution level,
                    deepest first). Their outputs will be used by aux heads.
        """
        self._intermediate_features = []

        def _make_hook(idx):
            def _hook(module, input, output):
                if len(self._intermediate_features) <= idx:
                    self._intermediate_features.append(output)
                else:
                    self._intermediate_features[idx] = output
            return _hook

        for i, layer in enumerate(layers):
            layer.register_forward_hook(_make_hook(i))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List of predictions: [main_output, aux_1, aux_2, ...].
            Main output is at full resolution; aux outputs are upsampled
            to match the target (full) spatial size.
        """
        self._intermediate_features = []

        main_output = self.base_model(x)
        if isinstance(main_output, tuple):
            main_output = main_output[0]

        target_size = main_output.shape[2:]
        outputs = [main_output]

        for i, (feat, head) in enumerate(zip(self._intermediate_features, self.aux_heads)):
            aux_pred = head(feat, target_size)
            outputs.append(aux_pred)

        return outputs

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Single-output inference (ignores auxiliary heads)."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs[0]


class DeepSupervisionUNet(nn.Module):
    """Self-contained U-Net with built-in deep supervision.

    A simpler alternative when you don't want to use hooks: this model
    wraps an encoder and builds a decoder with auxiliary heads inline.

    Args:
        encoder: Encoder that returns a list of skip features [s1, s2, ..., bottleneck].
        decoder_channels: Channel sizes for decoder levels.
        num_classes: Segmentation output channels.
        deep_supervision: Enable auxiliary outputs during training.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder_channels: List[int] = (256, 128, 64, 32),
        num_classes: int = 3,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.deep_supervision = deep_supervision

        # Decoder blocks: upsample + conv
        self.up_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for i, ch in enumerate(decoder_channels):
            in_ch = decoder_channels[i - 1] if i > 0 else decoder_channels[0] * 2
            skip_ch = ch  # assumes skip connection has same channels as decoder output
            self.up_blocks.append(
                nn.ConvTranspose3d(in_ch, ch, kernel_size=2, stride=2, bias=False)
            )
            self.conv_blocks.append(nn.Sequential(
                nn.Conv3d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(32, ch), ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(32, ch), ch),
                nn.LeakyReLU(inplace=True),
            ))

        # Main segmentation head
        self.seg_head = nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1)

        # Auxiliary heads for deep supervision
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(ch, num_classes, kernel_size=1)
                for ch in decoder_channels[:-1]
            ])

    def forward(self, x: torch.Tensor):
        # Encoder
        skips = self.encoder(x)
        if isinstance(skips, tuple):
            skips = list(skips)

        # skips: [s1 (highest res), s2, s3, s4, bottleneck (lowest res)]
        x = skips[-1]
        skip_features = skips[:-1][::-1]  # reverse to match decoder order

        aux_outputs = []
        for i, (up, conv) in enumerate(zip(self.up_blocks, self.conv_blocks)):
            x = up(x)
            if i < len(skip_features):
                skip = skip_features[i]
                # Handle size mismatch from non-power-of-2 dimensions
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = conv(x)

            # Capture for deep supervision (skip the last level — that's the main head)
            if self.deep_supervision and self.training and i < len(self.aux_heads):
                aux_outputs.append(x)

        main_output = self.seg_head(x)

        if self.deep_supervision and self.training and aux_outputs:
            target_size = main_output.shape[2:]
            aux_preds = [
                F.interpolate(
                    head(feat), size=target_size, mode="trilinear", align_corners=False
                )
                for head, feat in zip(self.aux_heads, aux_outputs)
            ]
            return [main_output] + aux_preds

        return main_output


def deep_supervision_loss(
    outputs: List[torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Compute weighted sum of losses across deep supervision outputs.

    Args:
        outputs: List of predictions [main, aux1, aux2, ...].
        target: Ground truth [B, C, D, H, W].
        criterion: Loss function (e.g., DiceFocalLoss).
        weights: Per-output weights. Default: exponentially decaying
                 [1.0, 0.5, 0.25, 0.125, ...].

    Returns:
        Weighted total loss.
    """
    n = len(outputs)
    if weights is None:
        weights = [1.0 / (2 ** i) for i in range(n)]

    # Normalize weights
    total_w = sum(weights[:n])
    weights = [w / total_w for w in weights[:n]]

    total_loss = torch.tensor(0.0, device=outputs[0].device)
    for output, w in zip(outputs, weights):
        total_loss = total_loss + w * criterion(output, target)

    return total_loss
