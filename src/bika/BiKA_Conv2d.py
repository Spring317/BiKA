import math
import torch

from torch import nn
from typing import Union, Tuple

from .functional import _BiKAConv2dFn


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

        self.reset_parameters()

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

        return _BiKAConv2dFn.apply(
            x,
            self.weight,
            self.bias,
            ph,
            pw,
            sh,
            sw,
        )


__all__ = [
    "BiKA_Conv2d",
]