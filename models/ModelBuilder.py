import torch
import torch.nn as nn
from torch.optim.lr_scheduler import PolynomialLR

from ranger import Ranger

from monai.networks.nets import UNet, SwinUNETR, DynUNet, SegResNet, SegResNetVAE
from monai.networks.nets.attentionunet import AttentionUnet

from .convnext import ConvNeXt
from .mednext import MedNeXt
from models import ResNet3dVAE, ResNet3d, nnUnet


# ---------------------------------------------------------------------------
# Model registry — add new architectures here without touching build_model()
# ---------------------------------------------------------------------------

def _build_segresnet(cfg):
    return SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
    )


def _build_segresnetvae(cfg):
    return SegResNetVAE(
        input_image_size=[128, 128, 128],
        init_filters=32,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
    )


def _build_convnext(cfg):
    return ConvNeXt(dims=[96, 192, 384, 768])


def _build_nnunext(cfg):
    kernels = [[3, 3, 3]] * 6
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    return DynUNet(
        spatial_dims=3,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        strides=strides,
        kernel_size=kernels,
        upsample_kernel_size=strides[1:],
        deep_supervision=False,
        deep_supr_num=2,
        act_name="GELU",
        res_block=True,
    )


def _build_nnunet(cfg):
    kernels = [[3, 3, 3]] * 6
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    return DynUNet(
        spatial_dims=3,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        strides=strides,
        kernel_size=kernels,
        upsample_kernel_size=strides[1:],
    )


def _build_custom_nnunet(cfg):
    return nnUnet(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"])


def _build_nunet(cfg):
    return UNet(
        spatial_dims=3,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2, 2),
        num_res_units=0,
        act="leakyrelu",
        norm=("layer", {"normalized_shape": (128,)}),
    )


def _build_unetr(cfg):
    return SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        depths=(2, 4, 2, 2),
    )


def _build_unet(cfg):
    return UNet(
        spatial_dims=3,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        act="RELU",
    )


def _build_attunet(cfg):
    return AttentionUnet(
        spatial_dims=3,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )


def _build_resnetvae(cfg):
    return ResNet3dVAE(
        in_channels=cfg["in_channels"],
        classes=cfg["num_classes"],
        dim=[128, 128, 128],
    )


def _build_resnet3d(cfg):
    return ResNet3d(
        in_channels=cfg["in_channels"],
        classes=cfg["num_classes"],
        dim=[128, 128, 128],
    )


def _build_mednext(cfg):
    return MedNeXt(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_channels=32,
        depth=2,
        kernel_size=3,
    )


def _build_mednext_large(cfg):
    return MedNeXt(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_channels=32,
        depth=3,
        kernel_size=5,
    )


MODEL_REGISTRY = {
    "SegResNet": _build_segresnet,
    "SegNetVAE": _build_segresnetvae,
    "ConvNeXt": _build_convnext,
    "nnUNeXt": _build_nnunext,
    "nnUNet": _build_nnunet,
    "nUNet": _build_nunet,
    "UNETR": _build_unetr,
    "UNet": _build_unet,
    "AttUnet": _build_attunet,
    "ResNetVAE": _build_resnetvae,
    "ResNet3d": _build_resnet3d,
    "CustomnnUNet": _build_custom_nnunet,
    "MedNeXt": _build_mednext,
    "MedNeXtLarge": _build_mednext_large,
}


def build_model(
    arch: str,
    num_classes: int,
    in_channels: int = 4,
    out_channels: int = 3,
    device: torch.device = torch.device("cuda:0"),
) -> nn.Module:
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    cfg = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_classes": num_classes,
    }
    model = MODEL_REGISTRY[arch](cfg).to(device)
    return model


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module, optim: str, cfg) -> torch.optim.Optimizer:
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)
    if optim == "ranger":
        return Ranger(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    raise ValueError(f"Unknown optimizer '{optim}'")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_scheduler(optimizer: torch.optim.Optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs, eta_min=0
        )
    if cfg.scheduler == "cycle":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg.lr,
            max_lr=cfg.max_lr,
            step_size_up=cfg.step_size_up,
            step_size_down=cfg.step_size_down,
            mode="triangular2",
            cycle_momentum=False,
            base_momentum=0.85,
            max_momentum=0.95,
        )
    if cfg.scheduler == "poly":
        # PolynomialLR built into PyTorch >= 1.13
        return PolynomialLR(optimizer, total_iters=cfg.n_iters, power=0.9)
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=cfg.n_iters,
            pct_start=0.3,
        )
    raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")
