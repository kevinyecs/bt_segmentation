"""
Parameter-Efficient Fine-Tuning (PEFT) for 3D medical segmentation models.

Methods:
  - LoRA for Conv3d layers (not just Linear)
  - Adapter injection into encoder-decoder networks
  - Encoder freezing with trainable decoder
  - Partial fine-tuning (freeze all but last N layers)

Usage:
    from utils.peft import apply_lora, freeze_encoder, apply_adapters, print_trainable_params

    model = build_model("SegResNet", num_classes=3)

    # Option 1: LoRA — ~5-10% of original parameters
    apply_lora(model, rank=4, target_modules=["conv"])
    print_trainable_params(model)

    # Option 2: Freeze encoder, train only decoder + head
    freeze_encoder(model, encoder_prefix="encoder")

    # Option 3: Inject lightweight adapters
    apply_adapters(model, bottleneck_dim=16)

Memory savings (approximate, for a 30M param model):
    Full fine-tuning:   ~12 GB VRAM (batch=1, 128^3)
    LoRA (rank=4):      ~4 GB  (only LoRA params + activations in grad graph)
    Frozen encoder:     ~6 GB  (encoder activations still cached, but no grads)
    Adapters:           ~5 GB  (small adapter params + activations)
"""

import math
import logging
from typing import Optional, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA for Conv3d (and Linear)
# ---------------------------------------------------------------------------

class LoRAConv3d(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Conv3d.

    Decomposes the weight update as: W' = W + (B @ A) * (alpha / rank)
    where A has shape [rank, in_ch, k, k, k] and B has [out_ch, rank, 1, 1, 1].

    The original conv weights are frozen; only A and B are trained.
    """

    def __init__(self, original: nn.Conv3d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        out_ch = original.out_channels
        in_ch = original.in_channels // original.groups
        k = original.kernel_size

        # A: projects input channels down to rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_ch, *k))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: projects rank back up to output channels
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank, 1, 1, 1))
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward (frozen)
        out = self.original(x)

        # LoRA path: x -> conv_A -> conv_B (no bias, same stride/padding)
        lora_out = F.conv3d(
            x,
            self.lora_A,
            stride=self.original.stride,
            padding=self.original.padding,
            dilation=self.original.dilation,
            groups=self.original.groups,
        )
        lora_out = F.conv3d(lora_out, self.lora_B)
        return out + lora_out * self.scaling


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha

        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + lora_out * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Replace Conv3d and Linear layers in-place with LoRA-wrapped versions.

    Args:
        model: The model to modify.
        rank: LoRA rank (lower = fewer params, typically 2-16).
        alpha: LoRA scaling factor.
        target_modules: List of substrings to match layer names.
            Default: ["conv", "linear", "pwconv", "dwconv"].
            Pass ["conv"] to only wrap convolutions.

    Returns:
        The modified model (same object, modified in-place).
    """
    if target_modules is None:
        target_modules = ["conv", "linear", "pwconv", "dwconv"]

    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if not any(t in full_name.lower() for t in target_modules):
                continue

            if isinstance(child, nn.Conv3d) and child.kernel_size != (1, 1, 1):
                setattr(module, child_name, LoRAConv3d(child, rank=rank, alpha=alpha))
                replaced += 1
            elif isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1

    logger.info("Applied LoRA (rank=%d) to %d layers", rank, replaced)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA weights back into original layers for inference speedup.

    After merging, the model runs at the same speed as the original (no
    extra LoRA forward passes), but with the fine-tuned weights baked in.
    """
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRAConv3d):
                # Merge: W' = W + B @ A * scaling
                with torch.no_grad():
                    delta = F.conv3d(
                        child.lora_A.unsqueeze(0).permute(1, 0, 2, 3, 4),
                        child.lora_B,
                    ).squeeze(0)
                    # Reshape delta to match original weight shape
                    child.original.weight.data += delta.reshape_as(child.original.weight) * child.scaling
                setattr(module, child_name, child.original)
                merged += 1
            elif isinstance(child, LoRALinear):
                with torch.no_grad():
                    delta = child.lora_B @ child.lora_A
                    child.original.weight.data += delta * child.scaling
                setattr(module, child_name, child.original)
                merged += 1

    logger.info("Merged %d LoRA layers back into original weights", merged)
    return model


# ---------------------------------------------------------------------------
# Adapter injection
# ---------------------------------------------------------------------------

class BottleneckAdapter(nn.Module):
    """Lightweight adapter: down-project -> nonlinearity -> up-project + skip.

    Reduces dimensionality to `bottleneck_dim`, applies GELU, then projects
    back. The residual connection means at initialization (with zero up-proj)
    the adapter is an identity function.
    """

    def __init__(self, channels: int, bottleneck_dim: int = 16):
        super().__init__()
        self.down = nn.Conv3d(channels, bottleneck_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv3d(bottleneck_dim, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)  # identity at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def apply_adapters(
    model: nn.Module,
    bottleneck_dim: int = 16,
    after_norm: bool = True,
) -> nn.Module:
    """Inject BottleneckAdapters after GroupNorm/BatchNorm layers.

    Freezes all original parameters; only adapters are trainable.

    Args:
        model: Model to modify.
        bottleneck_dim: Adapter hidden dimension (lower = fewer params).
        after_norm: If True, insert adapters after normalization layers.

    Returns:
        The modified model.
    """
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    injected = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, (nn.GroupNorm, nn.BatchNorm3d, nn.InstanceNorm3d)):
                channels = child.num_channels if hasattr(child, 'num_channels') else child.num_features
                adapter = BottleneckAdapter(channels, bottleneck_dim)
                # Replace norm with norm + adapter
                combined = nn.Sequential(child, adapter)
                setattr(module, child_name, combined)
                injected += 1

    logger.info("Injected %d adapters (bottleneck=%d)", injected, bottleneck_dim)
    return model


# ---------------------------------------------------------------------------
# Encoder freezing / partial fine-tuning
# ---------------------------------------------------------------------------

def freeze_encoder(
    model: nn.Module,
    encoder_prefix: str = "encoder",
    freeze_norm: bool = False,
) -> nn.Module:
    """Freeze encoder parameters, leaving decoder + head trainable.

    Works with any model where the encoder is a named sub-module.

    Args:
        model: Model to modify.
        encoder_prefix: Name prefix of the encoder sub-module.
        freeze_norm: If True, also freeze BatchNorm/GroupNorm running stats.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(encoder_prefix):
            param.requires_grad = False
            frozen += 1

    if freeze_norm:
        for name, module in model.named_modules():
            if name.startswith(encoder_prefix) and isinstance(
                module, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)
            ):
                module.eval()

    logger.info(
        "Frozen %d encoder parameters (prefix='%s'). "
        "Trainable: %d params",
        frozen, encoder_prefix,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


def freeze_all_but_last_n(model: nn.Module, n: int = 2) -> nn.Module:
    """Freeze all parameters except the last N named sub-modules.

    Useful when you want to fine-tune only the decoder head or the last
    few blocks of the network.
    """
    # Get all top-level children
    children = list(model.named_children())
    trainable_names = {name for name, _ in children[-n:]}

    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        param.requires_grad = top_level in trainable_names

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable: %d / %d params (%.1f%%) — last %d modules unfrozen",
        trainable, total, 100 * trainable / total, n,
    )
    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
    """Print and return trainable vs total parameter counts.

    Returns:
        (trainable, total, percentage)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    return trainable, total, pct


def get_peft_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer that only updates trainable parameters.

    Uses a higher learning rate than full fine-tuning since fewer params
    are being updated.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
