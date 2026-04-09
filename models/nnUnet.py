import torch
import torch.nn as nn

from .BaseModel import BaseModel


class nnUnet(BaseModel):
    """
    Custom nnUNet-inspired architecture with proper skip connections.
    Encoder-decoder with U-Net topology (skip connections at each resolution).
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super(nnUnet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        return self.decoder(skips)

    def test(self):
        inp = torch.rand(1, 4, 128, 128, 128)
        out = self.forward(inp)
        assert out.shape == (1, 3, 128, 128, 128), out.shape
        print("nnUnet test OK!")


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 4):
        super(Encoder, self).__init__()
        c = [in_channels, 32, 64, 128, 256, 320]
        self.layer1 = nn.Sequential(
            CNA(in_channels=c[0], out_channels=c[1], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[1], out_channels=c[1], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )
        self.down1 = DownScale(in_channels=c[1], out_channels=c[1], stride=(2, 2, 2), kernel_size=(3, 3, 3))

        self.layer2 = nn.Sequential(
            CNA(in_channels=c[1], out_channels=c[2], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[2], out_channels=c[2], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )
        self.down2 = DownScale(in_channels=c[2], out_channels=c[2], stride=(2, 2, 2), kernel_size=(3, 3, 3))

        self.layer3 = nn.Sequential(
            CNA(in_channels=c[2], out_channels=c[3], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[3], out_channels=c[3], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )
        self.down3 = DownScale(in_channels=c[3], out_channels=c[3], stride=(2, 2, 2), kernel_size=(3, 3, 3))

        self.layer4 = nn.Sequential(
            CNA(in_channels=c[3], out_channels=c[4], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[4], out_channels=c[4], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )
        self.down4 = DownScale(in_channels=c[4], out_channels=c[4], stride=(2, 2, 2), kernel_size=(3, 3, 3))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            CNA(in_channels=c[4], out_channels=c[5], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[5], out_channels=c[5], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )

    def forward(self, x: torch.Tensor):
        s1 = self.layer1(x)
        s2 = self.layer2(self.down1(s1))
        s3 = self.layer3(self.down2(s2))
        s4 = self.layer4(self.down3(s3))
        bottleneck = self.bottleneck(self.down4(s4))
        return s1, s2, s3, s4, bottleneck


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3):
        super(Decoder, self).__init__()
        c = [320, 256, 128, 64, 32]

        self.up4 = UpScale(in_channels=c[0], out_channels=c[1], stride=(2, 2, 2), kernel_size=(2, 2, 2))
        self.layer4 = nn.Sequential(
            CNA(in_channels=c[1] * 2, out_channels=c[1], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[1], out_channels=c[1], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )

        self.up3 = UpScale(in_channels=c[1], out_channels=c[2], stride=(2, 2, 2), kernel_size=(2, 2, 2))
        self.layer3 = nn.Sequential(
            CNA(in_channels=c[2] * 2, out_channels=c[2], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[2], out_channels=c[2], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )

        self.up2 = UpScale(in_channels=c[2], out_channels=c[3], stride=(2, 2, 2), kernel_size=(2, 2, 2))
        self.layer2 = nn.Sequential(
            CNA(in_channels=c[3] * 2, out_channels=c[3], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[3], out_channels=c[3], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )

        self.up1 = UpScale(in_channels=c[3], out_channels=c[4], stride=(2, 2, 2), kernel_size=(2, 2, 2))
        self.layer1 = nn.Sequential(
            CNA(in_channels=c[4] * 2, out_channels=c[4], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
            CNA(in_channels=c[4], out_channels=c[4], stride=(1, 1, 1), kernel_size=(3, 3, 3)),
        )

        self.out_conv = nn.Conv3d(c[4], out_channels, kernel_size=1)

    def forward(self, skips):
        s1, s2, s3, s4, x = skips

        x = self.up4(x)
        x = self.layer4(torch.cat([x, s4], dim=1))

        x = self.up3(x)
        x = self.layer3(torch.cat([x, s3], dim=1))

        x = self.up2(x)
        x = self.layer2(torch.cat([x, s2], dim=1))

        x = self.up1(x)
        x = self.layer1(torch.cat([x, s1], dim=1))

        return self.out_conv(x)


class CNA(nn.Module):
    """Convolution -> GroupNorm -> LeakyReLU"""
    def __init__(self, in_channels: int, out_channels: int, stride, kernel_size):
        super(CNA, self).__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        num_groups = min(32, out_channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DownScale(nn.Module):
    """Strided convolution for spatial downsampling."""
    def __init__(self, in_channels: int, out_channels: int, stride, kernel_size):
        super(DownScale, self).__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpScale(nn.Module):
    """Transposed convolution for spatial upsampling."""
    def __init__(self, in_channels: int, out_channels: int, stride, kernel_size):
        super(UpScale, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
