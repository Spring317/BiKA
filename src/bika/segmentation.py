import torch
import torch.nn.functional as F
from torch import nn

from .BiKA_Conv2d import BiKA_Conv2d

import torch.utils.checkpoint as checkpoint


# ── RPReLU (ReActNet, ECCV 2020) ─────────────────────────────────────────────
# Learnable activation that reshapes post-binarization distributions.
# At inference: can be fused into comparison + shift → multiply-free on ARM.
# ─────────────────────────────────────────────────────────────────────────────

class RPReLU(nn.Module):
    """ReActNet PReLU with learnable per-channel shifts and negative slope.
    
    y = ReLU(x + shift1) + slope_neg * min(x + shift1, 0) + shift2
    
    At inference, shift1/shift2 fuse into BN bias, slope_neg fuses into
    a conditional select (comparison, not multiplication).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.shift1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.slope_neg = nn.Parameter(torch.full((1, channels, 1, 1), 0.25))
        self.shift2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.shift1
        out = F.relu(x) - self.slope_neg * F.relu(-x) + self.shift2
        return out


# ── Channel-Adaptive Bypass (BiDense, 2024) ──────────────────────────────────
# Lightweight channel attention that lets critical channels bypass binarization.
# Cost: per-channel (not per-pixel) FP32, negligible on ARM.
# ─────────────────────────────────────────────────────────────────────────────

class ChannelBypass(nn.Module):
    """Squeeze-Excitation style gate that blends binary output with FP32 skip."""
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        mid = max(channels // ratio, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, binary_out: torch.Tensor, fp_skip: torch.Tensor) -> torch.Tensor:
        g = self.gate(fp_skip).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return binary_out * (1.0 - g) + fp_skip * g


# ── Enhanced BiKA Conv Block ─────────────────────────────────────────────────
# Incorporates all paper-derived fixes:
#   - RPReLU (ReActNet)       → learnable activation reshaping
#   - Channel Bypass (BiDense) → selective FP32 channel preservation
#   - FP32 Skip (Bi-Real Net)  → gradient highway
#   - Activation Checkpointing → VRAM reduction (~60% less activation memory)
# ─────────────────────────────────────────────────────────────────────────────

class BiKAConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Main binary path (BiKA_Conv2d already has Libra-PB + AdaBin)
        self.conv1 = BiKA_Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = RPReLU(out_ch)

        self.conv2 = BiKA_Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = RPReLU(out_ch)

        # FP32 Skip Connection (Bi-Real Net, ECCV 2018)
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()

        # Channel-Adaptive Bypass (BiDense, 2024)
        self.channel_bypass = ChannelBypass(out_ch)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Channel bypass: blend binary output with FP32 skip
        out = self.channel_bypass(out, skip)

        return self.act2(out + skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation checkpointing: trades ~30% more compute for ~60% less
        # activation VRAM by recomputing forward during backward.
        if self.training:
            return checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class BiKASegNet(nn.Module):
    """U-Net-style segmentation model built from BiKA conv blocks.

    Following standard binary-network practice (BNN/XNOR-Net), the first
    (stem) and last (classifier) layers are kept full-precision by default:
    binarizing the RGB stem discards input intensity information, and a
    BiKA classifier head can only emit even-integer logits in
    [-base_channels, base_channels], which is far too coarse for dense
    multi-class prediction.

    Incorporates the following paper-derived enhancements:
      - Libra-PB (IR-Net, CVPR 2020)      — in BiKA_Conv2d
      - AdaBin (ECCV 2022)                 — in BiKA_Conv2d
      - Error Decay Estimator (IR-Net)     — in CUDA backward kernel
      - RPReLU (ReActNet, ECCV 2020)       — in BiKAConvBlock
      - Channel Bypass (BiDense, 2024)     — in BiKAConvBlock
      - FP32 Skip (Bi-Real Net, ECCV 2018) — in BiKAConvBlock
      - Activation Checkpointing           — in BiKAConvBlock (VRAM reduction)

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
                RPReLU(base_channels),
                BiKA_Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                RPReLU(base_channels),
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
