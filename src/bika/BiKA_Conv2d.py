import math
import torch

from torch import nn
from typing import Union, Tuple

from .functional import bika_conv2d


class BiKA_Conv2d(nn.Module):
    """
    Supports stride >= 1, padding >= 0, dilation = 1, groups = 1.

    weight shape:
        (out_channels, in_channels, kernel_h, kernel_w)

    bias shape:
        (out_channels, in_channels, kernel_h, kernel_w)

    This is a per-connection bias design.
    """

    __constants__ = (
        "in_channels",
        "out_channels",
        "kernel_size",
        "padding",
        "stride",
        "per_connection_bias",
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if isinstance(padding, int):
            ph = pw = padding
        else:
            ph, pw = padding

        if isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride

        if dilation != 1:
            raise NotImplementedError(
                "BiKA_Conv2d currently supports only dilation=1"
            )

        if groups != 1:
            raise NotImplementedError(
                "BiKA_Conv2d currently supports only groups=1"
            )

        if ph < 0 or pw < 0:
            raise ValueError("padding must be >= 0")

        if sh < 1 or sw < 1:
            raise ValueError("stride must be >= 1")

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (int(kh), int(kw))
        self.padding = (int(ph), int(pw))
        self.stride = (int(sh), int(sw))
        self.per_connection_bias = True

        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels, kh, kw),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels, kh, kw),
                    **factory_kwargs,
                )
            )
        else:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_channels, in_channels, kh, kw),
                    **factory_kwargs,
                ),
                persistent=False,
            )

        self.register_buffer("out_scale", None, persistent=False)
        self.register_buffer("out_shift", None, persistent=False)
        self.do_relu = False

        # AdaBin (ECCV 2022): learnable per-channel output scaling.
        # At inference, fused into BatchNorm gamma — zero extra cost.
        self.output_scale = nn.Parameter(torch.ones(out_channels, 1, 1))

        self.reset_parameters()
        self.register_buffer("packed_weight", None)
        self.register_buffer("neg_bias_half", None)
        self.register_buffer("packed_bias_i8", None)
        # True bitpacked XNOR+POPCNT buffers (CPU)
        self.register_buffer("packed_w_bits", None)
        self.register_buffer("neg_bias_flat", None)
        
    def pack_weights(self):
        """Pre-packs float32 weights into int32 buffer for lightning-fast CUDA loading.
        Also pre-computes negated bias in half-precision (CUDA) and int8 (CPU AVX2)."""
        O, C, K, _ = self.weight.shape
        w_centered = self.weight - self.weight.mean(dim=[1, 2, 3], keepdim=True)
        w_flat = w_centered.view(O, C * K * K) >= 0.0
        num_words = (C * K * K + 31) // 32
        packed = torch.zeros((O, num_words), dtype=torch.int32, device=self.weight.device)
        for i in range(C * K * K):
            word_idx = i // 32
            bit_idx = i % 32
            bit_val = w_flat[:, i].to(torch.int32)
            packed[:, word_idx] |= (bit_val << bit_idx)
        self.packed_weight = packed
        # Pre-convert bias to half precision for CUDA
        self.neg_bias_half = self.bias.detach().half()
        # Pre-quantize bias to int8 (scale = 64.0) for CPU AVX2 SIMD kernel
        nb_i8 = torch.clamp(torch.round(self.bias.detach() * 64.0), -128, 127).to(torch.int8)
        self.packed_bias_i8 = nb_i8.contiguous()

        # ── True Bitpacked XNOR+POPCNT buffers (CPU) ──
        # Pack binary weight signs into uint64_t arrays (stored as int64 in PyTorch)
        CKK = C * K * K
        CKK_padded = ((CKK + 7) // 8) * 8  # round up to multiple of 8 for AVX2
        num_u64 = (CKK_padded + 63) // 64

        # Weight bits: 1 where centered weight >= 0, 0 otherwise
        # Use vectorized packing (much faster than per-bit loop)
        w_bits = w_flat.to(torch.int64)  # (O, CKK) with values {0, 1}
        packed_bits = torch.zeros((O, num_u64), dtype=torch.int64, device=self.weight.device)
        for i in range(CKK):
            u64_idx = i // 64
            bit_pos = i % 64
            packed_bits[:, u64_idx] |= (w_bits[:, i] << bit_pos)
        self.packed_w_bits = packed_bits

        # Flatten and pad bias to (O, CKK_padded) for aligned AVX2 loads
        # Padding values = +inf so comparison (0 >= inf) always fails → bit = 0
        nb_flat = self.bias.detach().view(O, CKK)
        if CKK_padded > CKK:
            padding = torch.full((O, CKK_padded - CKK), float('inf'),
                                 device=self.weight.device)
            nb_flat = torch.cat([nb_flat, padding], dim=1)
        self.neg_bias_flat = nb_flat.contiguous()

    def reset_parameters(self):
        fan_in = (
            self.in_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
        )

        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if isinstance(self.bias, nn.Parameter):
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                f"BiKA_Conv2d: expected x shape "
                f"(B, {self.in_channels}, H, W), got {tuple(x.shape)}"
            )

        ph, pw = self.padding
        sh, sw = self.stride

        # If packed_weight is available, skip redundant centering calculation
        if self.packed_weight is None:
            w = self.weight - self.weight.mean(dim=[1, 2, 3], keepdim=True)
        else:
            w = self.weight

        y = bika_conv2d(
            x,
            w,
            self.bias,
            out_scale=self.out_scale,
            out_shift=self.out_shift,
            packed_weight=self.packed_weight,
            neg_bias_half=self.neg_bias_half,
            packed_bias_i8=self.packed_bias_i8,
            packed_w_bits=self.packed_w_bits,
            neg_bias_flat=self.neg_bias_flat,
            do_relu=self.do_relu,
            padding=(ph, pw),
            stride=(sh, sw),
        )

        # AdaBin: per-channel scaling (fused into BN at inference)
        if self.output_scale is not None:
            return y * self.output_scale
        return y


__all__ = [
    "BiKA_Conv2d",
]