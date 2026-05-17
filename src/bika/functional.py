import torch
from typing import Union, Tuple

from . import _C


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
    def forward(ctx, x, w, b, pad_h: int, pad_w: int, stride_h: int, stride_w: int):
        y = _C.bika_conv2d_forward(
            x,
            w,
            b,
            int(pad_h),
            int(pad_w),
            int(stride_h),
            int(stride_w),
        )

        ctx.save_for_backward(x, w, b)
        ctx.pad_h = int(pad_h)
        ctx.pad_w = int(pad_w)
        ctx.stride_h = int(stride_h)
        ctx.stride_w = int(stride_w)

        return y

    @staticmethod
    def backward(ctx, gy):
        x, w, b = ctx.saved_tensors

        gi, gw, gb = _C.bika_conv2d_backward(
            gy.contiguous(),
            x,
            w,
            b,
            ctx.pad_h,
            ctx.pad_w,
            ctx.stride_h,
            ctx.stride_w,
        )

        return gi, gw, gb, None, None, None, None


def bika_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _BiKALinearFn.apply(x, w, b)


def bika_conv2d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> torch.Tensor:
    if isinstance(padding, int):
        ph = pw = padding
    else:
        ph, pw = padding

    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride

    return _BiKAConv2dFn.apply(
        x,
        w,
        b,
        int(ph),
        int(pw),
        int(sh),
        int(sw),
    )


__all__ = [
    "bika_linear",
    "bika_conv2d",
    "_BiKALinearFn",
    "_BiKAConv2dFn",
]