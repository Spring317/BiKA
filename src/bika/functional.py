import torch
from typing import Union, Tuple, Optional

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
    def forward(ctx, x, w, b, out_scale, out_shift, packed_weight, neg_bias_half, do_relu, pad_h: int, pad_w: int, stride_h: int, stride_w: int):
        # We need empty tensors if they are None
        if out_scale is None: out_scale = torch.empty(0, dtype=torch.float32, device=x.device)
        if out_shift is None: out_shift = torch.empty(0, dtype=torch.float32, device=x.device)
        if packed_weight is None: packed_weight = torch.empty(0, dtype=torch.int32, device=x.device)
        if neg_bias_half is None: neg_bias_half = torch.empty(0, dtype=torch.float16, device=x.device)
        
        y = _C.bika_conv2d_forward(
            x,
            w,
            b,
            out_scale,
            out_shift,
            packed_weight,
            neg_bias_half,
            bool(do_relu),
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

        return gi, gw, gb, None, None, None, None, None, None, None, None, None


def bika_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda:
        # Pure PyTorch CPU fallback
        x_reshaped = x.unsqueeze(1)  # [B, 1, I]
        b_reshaped = b.unsqueeze(0)  # [1, O, I]
        w_reshaped = w.unsqueeze(0)  # [1, O, I]
        x_bin = torch.where(x_reshaped + b_reshaped >= 0, 1.0, -1.0)
        w_bin = torch.where(w_reshaped >= 0, 1.0, -1.0)
        return (x_bin * w_bin).sum(dim=2)

    return _BiKALinearFn.apply(x, w, b)


def bika_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    out_scale: Optional[torch.Tensor] = None,
    out_shift: Optional[torch.Tensor] = None,
    packed_weight: Optional[torch.Tensor] = None,
    neg_bias_half: Optional[torch.Tensor] = None,
    packed_bias_i8: Optional[torch.Tensor] = None,
    packed_w_bits: Optional[torch.Tensor] = None,
    neg_bias_flat: Optional[torch.Tensor] = None,
    do_relu: bool = False,
) -> torch.Tensor:
    if isinstance(padding, int):
        ph = pw = padding
    else:
        ph, pw = padding

    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride

    if not input.is_cuda:
        # ── True Bitpacked XNOR+POPCNT kernel (preferred on CPU) ──
        if packed_w_bits is not None and neg_bias_flat is not None:
            out_scale_t = out_scale if out_scale is not None else torch.empty(0, device=input.device, dtype=torch.float32)
            out_shift_t = out_shift if out_shift is not None else torch.empty(0, device=input.device, dtype=torch.float32)

            return _C.bika_conv2d_forward_cpu_bitpack(
                input.contiguous(),
                packed_w_bits,
                neg_bias_flat,
                out_scale_t,
                out_shift_t,
                do_relu,
                int(sh),
                int(sw),
                int(ph),
                int(pw),
            )

        # ── Fallback: Int8 AVX2 SIMD CPU kernel (V2) ──
        out_scale_t = out_scale if out_scale is not None else torch.empty(0, device=input.device, dtype=torch.float32)
        out_shift_t = out_shift if out_shift is not None else torch.empty(0, device=input.device, dtype=torch.float32)
        packed_weight_t = packed_weight if packed_weight is not None else torch.empty(0, device=input.device, dtype=torch.int32)
        packed_bias_i8_t = packed_bias_i8 if packed_bias_i8 is not None else torch.empty(0, device=input.device, dtype=torch.int8)

        return _C.bika_conv2d_forward_cpu_int8_v2(
            input.contiguous(),
            weight.contiguous(),
            bias if bias is not None else torch.zeros(weight.shape[0], device=input.device),
            out_scale_t,
            out_shift_t,
            packed_weight_t,
            packed_bias_i8_t,
            do_relu,
            int(ph),
            int(pw),
            int(sh),
            int(sw),
        )

    return _BiKAConv2dFn.apply(
        input,
        weight,
        bias,
        out_scale,
        out_shift,
        packed_weight,
        neg_bias_half,
        do_relu,
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