import torch
from typing import Union, Tuple

from . import _C

# The CUDA kernels are float32-only. Under torch.amp.autocast, upstream
# layers may hand us float16/bfloat16 tensors, so mark the autograd
# functions autocast-aware: inputs are cast back to float32 and autocast
# is disabled inside forward/backward.
try:
    _custom_fwd = torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    _custom_bwd = torch.amp.custom_bwd(device_type="cuda")
except (AttributeError, TypeError):  # older torch (< 2.4)
    _custom_fwd = torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    _custom_bwd = torch.cuda.amp.custom_bwd


class _BiKALinearFn(torch.autograd.Function):
    @staticmethod
    @_custom_fwd
    def forward(ctx, x, w, b):
        y = _C.bika_linear_forward(x, w, b)
        ctx.save_for_backward(x, w, b)
        return y

    @staticmethod
    @_custom_bwd
    def backward(ctx, gy):
        x, w, b = ctx.saved_tensors
        gi, gw, gb = _C.bika_linear_backward(gy.contiguous(), x, w, b)
        return gi, gw, gb


class _BiKAConv2dFn(torch.autograd.Function):
    @staticmethod
    @_custom_fwd
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
    @_custom_bwd
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