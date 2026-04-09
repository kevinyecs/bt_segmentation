"""
BaseModel with checkpointing, mixed-precision inference, and parameter utilities.
Adapted from: https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/basemodel/basemodel.py
"""

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class BaseModel(nn.Module, ABC):
    """Base class for all segmentation models.

    Provides:
    - Checkpoint save / restore (best-loss tracking included)
    - Mixed-precision inference via torch.cuda.amp
    - Parameter count utility
    """

    def __init__(self):
        super().__init__()
        self.best_loss = float("inf")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def test(self):
        """Smoke-test: run a forward pass with a dummy tensor and assert shapes."""
        pass

    # ------------------------------------------------------------------
    # Device helper
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def restore_checkpoint(self, ckpt_file: str, optimizer=None) -> int:
        """Load weights from a .pth checkpoint.

        Args:
            ckpt_file: Path to the PyTorch checkpoint file.
            optimizer: If provided, optimizer state is also restored.

        Returns:
            The epoch at which the checkpoint was saved.
        """
        if not ckpt_file:
            raise ValueError("No checkpoint file specified.")

        try:
            ckpt_dict = torch.load(ckpt_file, map_location=self.device)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location="cpu")

        self.load_state_dict(ckpt_dict["model_state_dict"])

        if optimizer and ckpt_dict.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])

        return ckpt_dict["epoch"]

    def save_checkpoint(
        self,
        directory: str,
        epoch: int,
        loss: float,
        optimizer=None,
        name: str = None,
    ) -> None:
        """Save model (and optionally optimizer) state to disk.

        Always saves a 'last' checkpoint. If *loss* beats the current best,
        also saves a 'BEST' checkpoint.
        """
        os.makedirs(directory, exist_ok=True)

        ckpt_dict = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "loss": loss,
        }

        if name is None:
            name = f"{os.path.basename(directory)}_last_epoch.pth"

        torch.save(ckpt_dict, os.path.join(directory, name))

        if loss < self.best_loss:
            self.best_loss = loss
            best_name = f"{os.path.basename(directory)}_BEST.pth"
            torch.save(ckpt_dict, os.path.join(directory, best_name))

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def inference(self, input_tensor: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
        """Run a single forward pass in eval mode (no grad, optional AMP).

        Args:
            input_tensor: Input volume tensor.
            use_amp: Use automatic mixed precision (faster on supported GPUs).

        Returns:
            Prediction tensor on CPU.
        """
        self.eval()
        device = self.device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with autocast():
                    output = self.forward(input_tensor)
            else:
                output = self.forward(input_tensor)

        if isinstance(output, tuple):
            output = output[0]

        return output.cpu()

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------

    def count_params(self):
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
