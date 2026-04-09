"""
MedNeXt-style 3D segmentation architecture.

Inspired by: "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation"
(https://arxiv.org/abs/2303.09975)

Key ideas vs standard ConvNeXt:
  - Large depthwise kernels (3-7) for expanded receptive field without attention
  - Residual upsampling/downsampling blocks symmetric with the encoder
  - GroupNorm instead of LayerNorm (more stable for small 3D batch sizes)
  - Full encoder-decoder U-Net topology with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .BaseModel import BaseModel


class MedNeXtBlock(nn.Module):
    """Depthwise-separable residual block with large kernel and GELU activation."""

    def __init__(self, in_channels: int, expansion: int = 4, kernel_size: int = 3):
        super().__init__()
        mid = in_channels * expansion
        pad = kernel_size // 2

        self.dw_conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size, padding=pad, groups=in_channels, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.pw_expand = nn.Conv3d(in_channels, mid, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw_project = nn.Conv3d(mid, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_project(x)
        return x + residual


class MedNeXtDownBlock(nn.Module):
    """Downsampling block: strided conv that doubles channels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=pad, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class MedNeXtUpBlock(nn.Module):
    """Upsampling block: transposed conv that halves channels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class MedNeXt(BaseModel):
    """Full encoder-decoder MedNeXt with skip connections.

    Args:
        in_channels: Number of input modalities (default 4 for BraTS).
        out_channels: Number of segmentation classes (default 3 for TC/WT/ET).
        base_channels: Feature channels at the first encoder stage.
        depth: Number of MedNeXtBlocks per encoder/decoder stage.
        kernel_size: Depthwise conv kernel size (3, 5, or 7 — larger = bigger receptive field).
        expansion: Channel expansion factor inside each block.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 2,
        kernel_size: int = 3,
        expansion: int = 4,
    ):
        super().__init__()
        c = [base_channels * (2 ** i) for i in range(5)]  # [32, 64, 128, 256, 512]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c[0], kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, c[0]), num_channels=c[0]),
        )

        # Encoder
        self.enc1 = self._make_stage(c[0], depth, kernel_size, expansion)
        self.down1 = MedNeXtDownBlock(c[0], c[1])

        self.enc2 = self._make_stage(c[1], depth, kernel_size, expansion)
        self.down2 = MedNeXtDownBlock(c[1], c[2])

        self.enc3 = self._make_stage(c[2], depth, kernel_size, expansion)
        self.down3 = MedNeXtDownBlock(c[2], c[3])

        self.enc4 = self._make_stage(c[3], depth, kernel_size, expansion)
        self.down4 = MedNeXtDownBlock(c[3], c[4])

        # Bottleneck
        self.bottleneck = self._make_stage(c[4], depth, kernel_size, expansion)

        # Decoder
        self.up4 = MedNeXtUpBlock(c[4], c[3])
        self.dec4 = self._make_stage(c[3] * 2, depth, kernel_size, expansion)
        self.proj4 = nn.Conv3d(c[3] * 2, c[3], kernel_size=1, bias=False)

        self.up3 = MedNeXtUpBlock(c[3], c[2])
        self.dec3 = self._make_stage(c[2] * 2, depth, kernel_size, expansion)
        self.proj3 = nn.Conv3d(c[2] * 2, c[2], kernel_size=1, bias=False)

        self.up2 = MedNeXtUpBlock(c[2], c[1])
        self.dec2 = self._make_stage(c[1] * 2, depth, kernel_size, expansion)
        self.proj2 = nn.Conv3d(c[1] * 2, c[1], kernel_size=1, bias=False)

        self.up1 = MedNeXtUpBlock(c[1], c[0])
        self.dec1 = self._make_stage(c[0] * 2, depth, kernel_size, expansion)
        self.proj1 = nn.Conv3d(c[0] * 2, c[0], kernel_size=1, bias=False)

        self.out_conv = nn.Conv3d(c[0], out_channels, kernel_size=1)

        self._out_channels = out_channels
        self._in_channels = in_channels

    @staticmethod
    def _make_stage(channels: int, depth: int, kernel_size: int, expansion: int) -> nn.Sequential:
        return nn.Sequential(
            *[MedNeXtBlock(channels, expansion=expansion, kernel_size=kernel_size) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        d4 = self.proj4(self.dec4(torch.cat([self.up4(b), e4], dim=1)))
        d3 = self.proj3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)))
        d2 = self.proj2(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.proj1(self.dec1(torch.cat([self.up1(d2), e1], dim=1)))

        return self.out_conv(d1)

    def test(self):
        inp = torch.rand(1, self._in_channels, 128, 128, 128)
        out = self.forward(inp)
        assert out.shape == (1, self._out_channels, 128, 128, 128), out.shape
        print("MedNeXt test OK!")
