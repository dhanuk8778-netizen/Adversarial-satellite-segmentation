"""
U-Net for multispectral semantic segmentation.

Designed for 6-band Sentinel-2-style input (e.g. B2, B3, B4, B8, B11, B12 —
RGB + NIR + two SWIR bands) and dense 6-class land-cover prediction
(water, forest, urban, agriculture, barren, wetland).

Reference: Ronneberger et al., 2015, "U-Net: Convolutional Networks for
Biomedical Image Segmentation." Adapted here with batch norm, bilinear
upsampling (default) or transposed convs, and configurable depth/width so
the same class can be used for quick experiments (few filters) or the
full-capacity model used for the reported results.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling then double conv, with skip connection concat."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # pad x1 to match x2 spatial dims (handles odd input sizes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Configurable U-Net.

    Args:
        in_channels: number of spectral bands (6 for the Sentinel-2 setup
            used in this project).
        num_classes: number of land-cover classes (6: water, forest, urban,
            agriculture, barren, wetland).
        base_filters: width of the first encoder stage; doubles at each
            downsampling step (base -> 2b -> 4b -> 8b -> 16b at the
            bottleneck). 64 is the "full" configuration used to reach the
            reported 63.8% mIoU; 16-32 is convenient for CPU smoke tests.
        bilinear: use bilinear upsampling (fewer params, no checkerboard
            artifacts) instead of transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 6,
        base_filters: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        b = base_filters
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16 // factor)

        self.up1 = Up(b * 16, b * 8 // factor, bilinear)
        self.up2 = Up(b * 8, b * 4 // factor, bilinear)
        self.up3 = Up(b * 4, b * 2 // factor, bilinear)
        self.up4 = Up(b * 2, b, bilinear)
        self.outc = OutConv(b, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


LAND_COVER_CLASSES = ["water", "forest", "urban", "agriculture", "barren", "wetland"]
