import torch
import torch.nn.functional as F
from torch import nn

from .BiKA_Conv2d import BiKA_Conv2d


class BiKAConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            BiKA_Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            BiKA_Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BiKASegNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 16,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = BiKAConvBlock(in_channels, base_channels)
        self.enc2 = BiKAConvBlock(base_channels, base_channels * 2)
        self.enc3 = BiKAConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = BiKAConvBlock(base_channels * 4, base_channels * 8)

        # Decoder blocks take concatenated features (upsampled + skip connection)
        self.dec3 = BiKAConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = BiKAConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = BiKAConvBlock(base_channels * 2 + base_channels, base_channels)

        self.final = BiKA_Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = self.enc1(x)
        t2 = self.enc2(self.pool(t1))
        t3 = self.enc3(self.pool(t2))

        out = self.bottleneck(self.pool(t3))

        out = F.interpolate(out, size=t3.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, t3], dim=1)  # Concatenate along channel dimension
        out = self.dec3(out)

        out = F.interpolate(out, size=t2.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, t2], dim=1)  # Concatenate along channel dimension
        out = self.dec2(out)

        out = F.interpolate(out, size=t1.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, t1], dim=1)  # Concatenate along channel dimension
        out = self.dec1(out)

        return self.final(out)


__all__ = [
    "BiKASegNet",
]
