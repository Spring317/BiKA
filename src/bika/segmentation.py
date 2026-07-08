import torch
import torch.nn.functional as F
from torch import nn

from .BiKA_Conv2d import BiKA_Conv2d

import torch.utils.checkpoint as checkpoint

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
        if self.training and x.requires_grad:
            return checkpoint.checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


class BiKASegNet(nn.Module):
    """U-Net-style segmentation model built from BiKA conv blocks.

    Following standard binary-network practice (BNN/XNOR-Net), the first
    (stem) and last (classifier) layers are kept full-precision by default:
    binarizing the RGB stem discards input intensity information, and a
    BiKA classifier head can only emit even-integer logits in
    [-base_channels, base_channels], which is far too coarse for dense
    multi-class prediction.

    BiKA thresholds (= -bias) are re-initialized to *bika_bias_init* so they
    spread across the post-BatchNorm+ReLU activation range (~[0, 3]). The
    layer default of +-1/sqrt(fan_in) bunches every threshold near zero,
    so almost all binary connections output a constant sign(w) and the
    network cannot learn (verified: with the default init the model cannot
    even overfit 10 images; with this init it can).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 16,
        full_precision_stem: bool = True,
        full_precision_head: bool = True,
        bika_bias_init: tuple = (-2.0, 0.5),
        # Larger-magnitude weight init improves backward flow to deep layers
        # (grad ∝ w) but tested worse on the 10-image overfit benchmark
        # (0.517 vs 0.560 mIoU) — keep disabled by default.
        bika_weight_init: tuple = None,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        if full_precision_stem:
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                BiKA_Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.enc1 = BiKAConvBlock(in_channels, base_channels)

        self.enc2 = BiKAConvBlock(base_channels, base_channels * 2)
        self.enc3 = BiKAConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = BiKAConvBlock(base_channels * 4, base_channels * 8)

        # Decoder blocks take concatenated features (upsampled + skip connection)
        self.dec3 = BiKAConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = BiKAConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = BiKAConvBlock(base_channels * 2 + base_channels, base_channels)

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
                    # BiKA forward uses only sign(w); the magnitude's sole
                    # effect is backward gain (grad_input and grad_bias are
                    # both ∝ w). The default ~1/sqrt(fan_in) init starves
                    # deep layers of gradient (observed: enc3/bottleneck
                    # frozen at init in v5), so init magnitude is chosen
                    # for gradient flow, with random signs.
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
