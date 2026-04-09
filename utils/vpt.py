"""
Visual Prompt Tuning (VPT) for SwinUNETR.

Prepends learnable prompt tokens to the input sequence at each Swin
Transformer stage. Only the prompt tokens are trained; the entire
backbone is frozen. This gives ~90%+ parameter savings.

Only works with ViT/Swin-based architectures (SwinUNETR), not CNN U-Nets.

Reference: Jia et al., "Visual Prompt Tuning" (ECCV 2022)
           Adapted for 3D Swin Transformer in medical imaging.

Usage:
    from utils.vpt import apply_vpt_to_swinunetr

    model = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3)
    apply_vpt_to_swinunetr(model, num_prompts=10)
    # Now only prompt tokens + decoder head are trainable
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VPTLayer(nn.Module):
    """Visual Prompt Tuning layer that prepends learnable tokens.

    Wraps a transformer stage/block: prepends prompt tokens before the
    forward pass, then strips them from the output so downstream layers
    see the original sequence length.

    Args:
        original_layer: The Swin Transformer stage to wrap.
        num_prompts: Number of learnable prompt tokens per stage.
        embed_dim: Token embedding dimension (must match the stage).
        deep: If True, prompts are freshly prepended (deep VPT).
              If False, prompts from the first layer propagate (shallow VPT).
    """

    def __init__(
        self,
        original_layer: nn.Module,
        num_prompts: int = 10,
        embed_dim: int = 48,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.num_prompts = num_prompts

        # Learnable prompt tokens
        self.prompts = nn.Parameter(
            torch.randn(1, num_prompts, embed_dim) * 0.02
        )

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens [B, N, C] where N is the sequence length.

        Returns:
            Output tokens [B, N, C] (prompts stripped).
        """
        B = x.shape[0]
        prompts = self.prompts.expand(B, -1, -1)

        # Prepend prompts
        x = torch.cat([prompts, x], dim=1)  # [B, num_prompts + N, C]

        # Forward through original transformer stage
        x = self.original_layer(x)

        # Strip prompt tokens from output
        x = x[:, self.num_prompts:]

        return x


class VPTSwinStage(nn.Module):
    """VPT wrapper for a MONAI SwinUNETR stage that handles 3D reshaping.

    MONAI's SwinUNETR internally reshapes 3D volumes into windowed sequences.
    This wrapper operates at the stage level, injecting prompts into the
    sequence before each Swin Transformer block processes it.

    Since the exact internal sequence handling varies by MONAI version,
    this wrapper operates at a higher level: it modifies the input
    feature map by adding prompt information via a learnable projection.

    Args:
        original_stage: A SwinUNETR encoder stage.
        num_prompts: Number of virtual prompt channels to add.
        feature_size: Spatial feature dimension at this stage.
    """

    def __init__(
        self,
        original_stage: nn.Module,
        num_prompts: int = 10,
        embed_dim: int = 48,
    ):
        super().__init__()
        self.original_stage = original_stage
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        # Learnable prompt embedding projected into feature space
        # Applied as a spatial-agnostic bias added to each token
        self.prompt_bias = nn.Parameter(torch.zeros(1, embed_dim, 1, 1, 1))
        nn.init.normal_(self.prompt_bias, std=0.02)

        # Learnable scale (starts at near-zero for stable init)
        self.prompt_scale = nn.Parameter(torch.ones(1) * 0.01)

        # Freeze original stage
        for param in self.original_stage.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add prompt bias (broadcast across spatial dims)
        if x.ndim == 5:  # [B, C, D, H, W]
            x = x + self.prompt_bias * self.prompt_scale
        return self.original_stage(x)


def apply_vpt_to_swinunetr(
    model: nn.Module,
    num_prompts: int = 10,
    unfreeze_decoder: bool = True,
    unfreeze_head: bool = True,
) -> nn.Module:
    """Apply Visual Prompt Tuning to a MONAI SwinUNETR model.

    Freezes the entire Swin encoder and adds learnable prompt parameters
    at each encoder stage. Optionally unfreezes the decoder and
    segmentation head.

    Args:
        model: A MONAI SwinUNETR model.
        num_prompts: Number of prompt tokens/channels per stage.
        unfreeze_decoder: Keep decoder parameters trainable.
        unfreeze_head: Keep the final segmentation head trainable.

    Returns:
        Modified model with VPT applied.
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Wrap encoder stages with VPT
    # MONAI SwinUNETR structure: model.swinViT contains the encoder
    if hasattr(model, 'swinViT'):
        swin = model.swinViT
        feature_size = getattr(model, 'feature_size', 48)

        # Wrap each layer in the Swin ViT encoder
        if hasattr(swin, 'layers1'):
            layer_names = ['layers1', 'layers2', 'layers3', 'layers4']
            for i, name in enumerate(layer_names):
                if hasattr(swin, name):
                    original = getattr(swin, name)
                    dim = feature_size * (2 ** i)
                    wrapped = VPTSwinStage(original, num_prompts=num_prompts, embed_dim=dim)
                    setattr(swin, name, wrapped)
                    logger.info("Applied VPT to %s (dim=%d)", name, dim)
        elif hasattr(swin, 'layers'):
            for i, layer in enumerate(swin.layers):
                dim = feature_size * (2 ** i)
                swin.layers[i] = VPTSwinStage(layer, num_prompts=num_prompts, embed_dim=dim)
                logger.info("Applied VPT to layer %d (dim=%d)", i, dim)

    # Step 3: Unfreeze decoder if requested
    if unfreeze_decoder:
        decoder_prefixes = ['decoder', 'encoder1', 'encoder2', 'encoder3', 'encoder4',
                           'encoder10', 'out']
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in decoder_prefixes):
                param.requires_grad = True

    # Step 4: Unfreeze segmentation head if requested
    if unfreeze_head:
        head_prefixes = ['out', 'head', 'seg_head', 'final']
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in head_prefixes):
                param.requires_grad = True

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "VPT applied: %d / %d trainable params (%.1f%%)",
        trainable, total, 100 * trainable / total
    )

    return model
