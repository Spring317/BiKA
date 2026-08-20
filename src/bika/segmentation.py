import torch
import torch.nn.functional as F
from torch import nn

from .BiKA_Conv2d import BiKA_Conv2d

import torch.utils.checkpoint as checkpoint


class RPReLU(nn.Module):
    """ReActNet-style PReLU with learnable pre/post shifts for BNNs."""
    def __init__(self, channels):
        super().__init__()
        self.shift_pre = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.prelu = nn.PReLU(channels)
        self.shift_post = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        return self.prelu(x + self.shift_pre) + self.shift_post


class BiKAConvBlock(nn.Module):
    """Enhanced BiKA block with paper-derived accuracy fixes.

    Active fixes (zero/negligible VRAM cost):
      - Libra-PB + AdaBin + EDE → inside BiKA_Conv2d / CUDA kernel
      - PReLU (ReActNet-inspired) → learnable negative slope, replaces ReLU
      - FP32 skip (Bi-Real Net) → gradient highway
      - Activation checkpointing → ~60% less VRAM during training
    """
    def __init__(self, in_ch: int, out_ch: int, legacy_mode: bool = False):
        super().__init__()
        self.legacy_mode = legacy_mode
        self.conv1 = BiKA_Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU(inplace=True) if legacy_mode else RPReLU(out_ch)

        self.conv2 = BiKA_Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU(inplace=True) if legacy_mode else RPReLU(out_ch)

        # FP32 Skip Connection (Bi-Real Net, ECCV 2018)
        if not legacy_mode and in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if self.legacy_mode:
            if isinstance(self.bn1, nn.Identity) and isinstance(self.act1, nn.Identity) and \
               isinstance(self.bn2, nn.Identity) and isinstance(self.act2, nn.Identity):
                return self.conv2(self.conv1(x))
            out = self.act1(self.bn1(self.conv1(x)))
            return self.act2(self.bn2(self.conv2(out)))
        else:
            skip = self.skip(x)
            out = self.act1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.act2(out + skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class BiKASegNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 16,
        full_precision_stem: bool = True,
        full_precision_head: bool = True,
        bika_bias_init: tuple = (-2.0, 0.5),
        bika_weight_init: tuple = None,
        legacy_mode: bool = False,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.legacy_mode = legacy_mode

        if full_precision_stem:
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True) if legacy_mode else RPReLU(base_channels),
                BiKA_Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True) if legacy_mode else RPReLU(base_channels),
            )
        else:
            self.enc1 = BiKAConvBlock(in_channels, base_channels, legacy_mode=legacy_mode)

        self.enc2 = BiKAConvBlock(base_channels, base_channels * 2, legacy_mode=legacy_mode)
        self.enc3 = BiKAConvBlock(base_channels * 2, base_channels * 4, legacy_mode=legacy_mode)
        self.bottleneck = BiKAConvBlock(base_channels * 4, base_channels * 8, legacy_mode=legacy_mode)

        self.dec3 = BiKAConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4, legacy_mode=legacy_mode)
        self.dec2 = BiKAConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, legacy_mode=legacy_mode)
        self.dec1 = BiKAConvBlock(base_channels * 2 + base_channels, base_channels, legacy_mode=legacy_mode)

        if full_precision_head:
            self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        else:
            self.final = BiKA_Conv2d(base_channels, num_classes, kernel_size=1)

        with torch.no_grad():
            for m in self.modules():
                if not isinstance(m, BiKA_Conv2d):
                    continue
                if bika_bias_init is not None and isinstance(m.bias, nn.Parameter):
                    low, high = bika_bias_init
                    m.bias.uniform_(low, high)
                if bika_weight_init is not None:
                    lo, hi = bika_weight_init
                    sign = torch.randint_like(m.weight, 0, 2) * 2 - 1
                    m.weight.copy_(
                        torch.empty_like(m.weight).uniform_(lo, hi) * sign
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = self.enc1(x)
        t2 = self.enc2(self.pool(t1))
        t3 = self.enc3(self.pool(t2))

        out = self.bottleneck(self.pool(t3))

        out = F.interpolate(out, size=t3.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, t3], dim=1)
        out = self.dec3(out)

        out = F.interpolate(out, size=t2.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, t2], dim=1)
        out = self.dec2(out)

        out = F.interpolate(out, size=t1.shape[2:], mode="bilinear", align_corners=False)
        out = torch.cat([out, t1], dim=1)
        out = self.dec1(out)

        return self.final(out)


__all__ = [
    "BiKASegNet",
]
