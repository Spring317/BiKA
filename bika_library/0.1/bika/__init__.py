import math
import torch
from torch import nn
from . import _C  # compiled CUDA extension

# -------- Autograd Functions (thin wrappers over CUDA) --------
class _BiKALinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        y = _C.bika_linear_forward(x, w, b)
        ctx.save_for_backward(x, w, b)
        return y
    @staticmethod
    def backward(ctx, gy):
        x, w, b = ctx.saved_tensors
        gi, gw, gb = _C.bika_linear_backward(gy.contiguous(), x, w, b)
        return gi, gw, gb

class _BiKAConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        y = _C.bika_conv2d_forward(x, w, b)
        ctx.save_for_backward(x, w, b)
        return y
    @staticmethod
    def backward(ctx, gy):
        x, w, b = ctx.saved_tensors
        gi, gw, gb = _C.bika_conv2d_backward(gy.contiguous(), x, w, b)
        return gi, gw, gb

# -------- Public functional APIs (optional) --------
def bika_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _BiKALinearFn.apply(x, w, b)

def bika_conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _BiKAConv2dFn.apply(x, w, b)

# -------- nn.Module wrappers (drop-in usage) --------
class BiKA_Linear(nn.Module):
    """
    Drop-in like nn.Linear, but:
      - weight: (out_features, in_features)
      - bias:   (out_features, in_features)  # per-connection bias (not shape [out_features])
    Forward computes sum_i sign((x_i + b_oi) * w_oi), no sign after the sum.
    """
    __constants__ = ("in_features", "out_features", "per_connection_bias")

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.per_connection_bias = True

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        else:
            # keep a zero bias buffer of matching shape so the CUDA kernel always receives a tensor
            self.register_buffer("bias", torch.zeros((out_features, in_features), **factory_kwargs), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # mimic nn.Linear's default (kaiming_uniform for weight, and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)))
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if isinstance(self.bias, nn.Parameter):
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features) -> y: (B, out_features)
        """
        if x.dim() != 2 or x.size(-1) != self.in_features:
            raise ValueError(f"BiKA_Linear: expected x shape (B,{self.in_features}), got {tuple(x.shape)}")
        # ensure contiguity/dtypes handled in CUDA kernels; parameters live on the same device as module.
        return _BiKALinearFn.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={isinstance(self.bias, nn.Parameter)} (per-connection)"


class BiKA_Conv2d(nn.Module):
    """
    Drop-in like nn.Conv2d (subset):
      - weight: (out_channels, in_channels, kH, kW)
      - bias:   (out_channels, in_channels, kH, kW)  # per-connection bias
    Current CUDA supports VALID conv only: stride=1, padding=0, dilation=1, groups=1.
    """
    __constants__ = ("in_channels", "out_channels", "kernel_size", "per_connection_bias")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
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
        if stride != 1 or padding != 0 or dilation != 1 or groups != 1:
            raise NotImplementedError("BiKA_Conv2d currently supports only stride=1, padding=0, dilation=1, groups=1")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.per_connection_bias = True

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kh, kw), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels, in_channels, kh, kw), **factory_kwargs))
        else:
            self.register_buffer("bias", torch.zeros((out_channels, in_channels, kh, kw), **factory_kwargs), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # match torch.nn.Conv2d default-ish init
        # kaiming_uniform on weight; bias uses fan_in bound (per-connection bias has same shape as weight)
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if isinstance(self.bias, nn.Parameter):
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) -> y: (B, O, H-kH+1, W-kW+1)  (valid conv)
        """
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(f"BiKA_Conv2d: expected x shape (B,{self.in_channels},H,W), got {tuple(x.shape)}")
        return _BiKAConv2dFn.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        kH, kW = self.kernel_size
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size=({kH}, {kW}), bias={isinstance(self.bias, nn.Parameter)} (per-connection), "
                "stride=1, padding=0, dilation=1, groups=1")

__all__ = [
    "bika_linear",
    "bika_conv2d",
    "BiKA_Linear",
    "BiKA_Conv2d",
]

