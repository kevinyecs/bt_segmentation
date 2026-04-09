"""
HuggingFace Hub and MONAI Model Zoo integration for loading pre-trained
3D medical segmentation models.

Supports:
  - MONAI Bundles (SwinUNETR-BraTS, SegResNet-BraTS, etc.)
  - HuggingFace Hub models (SegFormer3D, MedSAM, custom checkpoints)
  - Direct weight loading from URLs or local paths

Usage:
    from utils.hub import load_monai_bundle, load_hf_model, load_pretrained_weights

    # MONAI bundle (downloads + loads architecture + weights)
    model = load_monai_bundle("brats_mri_segmentation")

    # HuggingFace Hub checkpoint
    model = load_hf_model("pnavard/SegFormer3D", num_classes=3)

    # Load pre-trained weights into any existing model
    model = build_model("SegResNet", num_classes=3)
    load_pretrained_weights(model, "path/to/checkpoint.pth")
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MONAI Model Zoo / Bundle integration
# ---------------------------------------------------------------------------

# Known MONAI bundles for brain tumor segmentation
MONAI_BUNDLES = {
    "brats_mri_segmentation": {
        "name": "brats_mri_segmentation",
        "description": "SegResNet trained on BraTS (Medical Segmentation Decathlon Task01)",
        "arch": "SegResNet",
    },
    "swinunetr_btcv_segmentation": {
        "name": "swinunetr_btcv_segmentation",
        "description": "SwinUNETR pre-trained on BTCV, transferable to BraTS",
        "arch": "SwinUNETR",
    },
    "wholeBody_ct_segmentation": {
        "name": "wholeBody_ct_segmentation",
        "description": "Whole-body CT segmentation (STU-Net style)",
        "arch": "SegResNet",
    },
}


def list_available_bundles() -> dict:
    """Return the dictionary of known MONAI bundles for reference."""
    return MONAI_BUNDLES


def load_monai_bundle(
    bundle_name: str,
    bundle_dir: str = "./bundles",
    device: str = "cuda",
    version: Optional[str] = None,
) -> nn.Module:
    """Download and load a MONAI bundle model with pre-trained weights.

    Requires: ``pip install "monai[fire,nibabel]"``

    Args:
        bundle_name: Name of the bundle in MONAI Model Zoo.
        bundle_dir: Local directory to cache downloaded bundles.
        device: Target device.
        version: Specific bundle version (None = latest).

    Returns:
        A PyTorch nn.Module with pre-trained weights loaded.
    """
    try:
        from monai.bundle import ConfigParser, download
    except ImportError as e:
        raise ImportError(
            "MONAI bundle support requires: pip install 'monai[fire,nibabel]'"
        ) from e

    bundle_path = Path(bundle_dir) / bundle_name

    if not bundle_path.exists():
        logger.info("Downloading MONAI bundle '%s' to %s ...", bundle_name, bundle_dir)
        download(
            name=bundle_name,
            bundle_dir=bundle_dir,
            version=version,
        )

    # Parse the bundle config and instantiate the network
    config_path = bundle_path / "configs" / "inference.json"
    if not config_path.exists():
        config_path = bundle_path / "configs" / "inference.yaml"

    parser = ConfigParser()
    parser.read_config(str(config_path))
    parser.parse()

    model = parser.get_parsed_content("network_def")

    # Load pre-trained weights
    weights_path = bundle_path / "models" / "model.pt"
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state)
        logger.info("Loaded pre-trained weights from %s", weights_path)
    else:
        logger.warning("No pre-trained weights found at %s", weights_path)

    return model.to(device)


# ---------------------------------------------------------------------------
# HuggingFace Hub integration
# ---------------------------------------------------------------------------

def load_hf_model(
    repo_id: str,
    filename: str = "pytorch_model.bin",
    model_class: Optional[type] = None,
    model_kwargs: Optional[dict] = None,
    device: str = "cuda",
    trust_remote_code: bool = False,
) -> nn.Module:
    """Download a model checkpoint from HuggingFace Hub and load it.

    For models that ship as raw PyTorch checkpoints (not transformers-native):

    Args:
        repo_id: HuggingFace repo ID (e.g., "pnavard/SegFormer3D").
        filename: Checkpoint filename inside the repo.
        model_class: The nn.Module class to instantiate. If None, attempts
                     to load the full checkpoint (state_dict + architecture).
        model_kwargs: Kwargs passed to model_class.__init__().
        device: Target device.
        trust_remote_code: Allow running remote code from the repo.

    Returns:
        A PyTorch nn.Module with weights loaded.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "HuggingFace Hub integration requires: pip install huggingface-hub"
        ) from e

    logger.info("Downloading %s/%s from HuggingFace Hub...", repo_id, filename)
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    checkpoint = torch.load(local_path, map_location=device)

    if model_class is not None:
        model_kwargs = model_kwargs or {}
        model = model_class(**model_kwargs)

        # Handle both raw state_dict and wrapped checkpoints
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded weights into %s (strict=False)", model_class.__name__)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        raise ValueError(
            f"Checkpoint is a dict but no model_class was provided. "
            f"Keys found: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
        )

    return model.to(device)


def load_hf_swinunetr_pretrained(
    in_channels: int = 4,
    out_channels: int = 3,
    img_size: tuple = (128, 128, 128),
    device: str = "cuda",
) -> nn.Module:
    """Load SwinUNETR with self-supervised pre-trained weights from HuggingFace.

    These weights are from the MONAI research team's pre-training on a large
    collection of CT/MRI scans, and can be fine-tuned for BraTS.
    """
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=48,
    )

    try:
        from huggingface_hub import hf_hub_download

        weight_path = hf_hub_download(
            repo_id="Project-MONAI/SwinUNETR",
            filename="model_swinvit.pt",
        )
        state = torch.load(weight_path, map_location=device)

        # The pre-trained weights are for the SwinViT encoder only
        if "state_dict" in state:
            state = state["state_dict"]

        # Filter to swinViT keys only
        swin_state = {
            k.replace("module.", ""): v
            for k, v in state.items()
            if "swinViT" in k or k.startswith("encoder") or k.startswith("layers")
        }

        model.load_state_dict(swin_state, strict=False)
        logger.info("Loaded SwinUNETR self-supervised pre-trained weights")
    except Exception as e:
        logger.warning("Could not load pre-trained SwinUNETR weights: %s", e)
        logger.info("Falling back to random initialization")

    return model.to(device)


# ---------------------------------------------------------------------------
# Generic weight loading
# ---------------------------------------------------------------------------

def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    key: Optional[str] = None,
) -> nn.Module:
    """Load weights from a local checkpoint file into an existing model.

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to .pth / .pt file.
        strict: Whether to require exact key matching.
        key: If the checkpoint is a dict, which key holds the state_dict
             (e.g., "model_state_dict", "state_dict"). Auto-detected if None.

    Returns:
        The model with loaded weights.
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if key is not None:
            state_dict = checkpoint[key]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Expected dict checkpoint, got {type(checkpoint)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.info("Missing keys (%d): %s...", len(missing), missing[:5])
    if unexpected:
        logger.info("Unexpected keys (%d): %s...", len(unexpected), unexpected[:5])

    return model
