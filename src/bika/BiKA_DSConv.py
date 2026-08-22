"""Depthwise-Separable BiKA Convolution Block.

Factorizes dense BiKA_Conv2d(C_in, C_out, 3×3) into:
  1. DW-BiKA:  BiKA_Conv2d(C_in, C_in, 3×3, groups=C_in)  — binary depthwise
  2. PW-FP32:  nn.Conv2d(C_in, C_out, 1×1)                 — FP32 pointwise

Operation count reduction:
  Dense:  O(C_out × C_in × K²)  per pixel   e.g. 32 × 32 × 9 = 9216
  DS:     O(C_in × K² + C_in × C_out)       e.g. 32 × 9 + 32 × 32 = 1312  (~7× fewer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiKA_DWConv2d(nn.Module):
    """Binary depthwise convolution: each channel is processed independently.
    
    weight shape: (channels, 1, K, K)
    bias shape:   (channels, 1, K, K)
    
    For each channel c and spatial position (h, w):
        output[c, h, w] = Σ_{kh,kw} sign(x[c, h+kh, w+kw] + bias[c, 0, kh, kw])
                                     * sign(w_centered[c, 0, kh, kw])
    
    CKK = 1 × K × K = 9 for K=3  (vs CKK = C × K × K = 288 for dense C=32)
    """
    
    def __init__(self, channels, kernel_size=3, padding=1, stride=1):
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
        
        self.channels = channels
        self.kernel_size = (kh, kw)
        self.padding = (ph, pw)
        self.stride = (sh, sw)
        
        # Binary weight: (C, 1, K, K)
        self.weight = nn.Parameter(torch.empty(channels, 1, kh, kw))
        # Per-connection bias (threshold): (C, 1, K, K)
        self.bias = nn.Parameter(torch.empty(channels, 1, kh, kw))
        
        # AdaBin scaling
        self.output_scale = nn.Parameter(torch.ones(channels, 1, 1))
        
        # Fused BN buffers
        self.register_buffer("out_scale", None, persistent=False)
        self.register_buffer("out_shift", None, persistent=False)
        self.do_relu = False
        
        # Precomputed packed buffers for C++ CPU kernel
        self.register_buffer("neg_bias_flat", None)
        self.register_buffer("w_sign_i8", None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.uniform_(self.bias, -2.0, 0.5)
    
    def pack_weights(self):
        """Precompute packed threshold and binarized weights for C++ CPU kernel."""
        with torch.no_grad():
            w = self.weight - self.weight.mean(dim=[1, 2, 3], keepdim=True)
            w_sign = torch.where(w >= 0, 1, -1).to(torch.int8)
            self.w_sign_i8 = w_sign.view(self.channels, -1).contiguous()
            
            # Threshold = -bias (consistent with CUDA & CPU kernels)
            self.neg_bias_flat = (-self.bias.detach()).view(self.channels, -1).contiguous().float()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"
        
        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride
        
        if not x.is_cuda:
            try:
                from . import _C
            except ImportError:
                import bika._C as _C
            
            if self.neg_bias_flat is None or self.w_sign_i8 is None:
                self.pack_weights()
            
            out_scale_t = self.out_scale if self.out_scale is not None else torch.empty(0, device=x.device, dtype=torch.float32)
            out_shift_t = self.out_shift if self.out_shift is not None else torch.empty(0, device=x.device, dtype=torch.float32)
            
            out = _C.bika_dw_conv2d_forward_cpu(
                x.contiguous(),
                self.neg_bias_flat,
                self.w_sign_i8,
                out_scale_t,
                out_shift_t,
                self.do_relu,
                int(ph), int(pw),
                int(sh), int(sw),
                int(kh), int(kw),
            )
            if self.output_scale is not None and self.out_scale is None:
                out = out * self.output_scale.unsqueeze(0)
            return out
        
        # GPU / Autograd fallback path
        w = self.weight - self.weight.mean(dim=[1, 2, 3], keepdim=True)
        w_sign = torch.where(w >= 0, 1.0, -1.0)
        neg_bias = -self.bias
        
        if ph > 0 or pw > 0:
            x = F.pad(x, [pw, pw, ph, ph])
        
        Ho = (H + 2 * ph - kh) // sh + 1
        Wo = (W + 2 * pw - kw) // sw + 1
        
        acc = torch.zeros(B, C, Ho, Wo, device=x.device, dtype=x.dtype)
        for ikh in range(kh):
            for ikw in range(kw):
                h_end = ikh + Ho * sh
                w_end = ikw + Wo * sw
                x_slice = x[:, :, ikh:h_end:sh, ikw:w_end:sw]
                thresh = neg_bias[:, 0, ikh, ikw].view(1, -1, 1, 1)
                w_s = w_sign[:, 0, ikh, ikw].view(1, -1, 1, 1)
                x_s = (x_slice >= thresh).to(x.dtype) * 2 - 1
                acc = acc + x_s * w_s
        
        if self.output_scale is not None:
            acc = acc * self.output_scale.unsqueeze(0)
        if self.out_scale is not None:
            acc = acc * self.out_scale.view(1, -1, 1, 1) + self.out_shift.view(1, -1, 1, 1)
        if self.do_relu:
            acc = F.relu(acc, inplace=True)
        
        return acc


class BiKA_DSConvBlock(nn.Module):
    """Depthwise-Separable BiKA block.
    
    Architecture:
        DW-BiKA(C_in, C_in, 3×3) → BN → ReLU →
        PW-FP32(C_in, C_out, 1×1) → BN → ReLU
    
    Optionally with skip connection (Bi-Real Net style).
    """
    
    def __init__(self, in_ch, out_ch, legacy_mode=False):
        super().__init__()
        self.legacy_mode = legacy_mode
        
        # Depthwise binary conv
        self.dw_conv = BiKA_DWConv2d(in_ch, kernel_size=3, padding=1)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_act = nn.ReLU(inplace=True)
        
        # Pointwise FP32 conv (channel mixing)
        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.dw_act(self.dw_bn(self.dw_conv(x)))
        out = self.pw_act(self.pw_bn(self.pw_conv(out)))
        return out


class BiKASegNet_DS(nn.Module):
    """Depthwise-Separable BiKA Segmentation Network.
    
    Same U-Net architecture as BiKASegNet but with DS blocks
    that are ~7-9× more compute-efficient per layer.
    """
    
    def __init__(
        self,
        num_classes=2,
        in_channels=3,
        base_channels=16,
        legacy_mode=True,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        bc = base_channels
        
        # FP32 stem (first layer must handle 3-channel RGB input)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bc),
            nn.ReLU(inplace=True),
            # Second conv in enc1: DS block (DW stays at bc channels)
            BiKA_DWConv2d(bc, kernel_size=3, padding=1),
            nn.BatchNorm2d(bc),
            nn.ReLU(inplace=True),
        )
        
        # Encoder: DS blocks with channel expansion via pointwise
        self.enc2 = BiKA_DSConvBlock(bc, bc * 2, legacy_mode=legacy_mode)
        self.enc3 = BiKA_DSConvBlock(bc * 2, bc * 4, legacy_mode=legacy_mode)
        self.bottleneck = BiKA_DSConvBlock(bc * 4, bc * 8, legacy_mode=legacy_mode)
        
        # Decoder: DS blocks with channel reduction
        self.dec3 = BiKA_DSConvBlock(bc * 8 + bc * 4, bc * 4, legacy_mode=legacy_mode)
        self.dec2 = BiKA_DSConvBlock(bc * 4 + bc * 2, bc * 2, legacy_mode=legacy_mode)
        self.dec1 = BiKA_DSConvBlock(bc * 2 + bc, bc, legacy_mode=legacy_mode)
        
        # FP32 head
        self.final = nn.Conv2d(bc, num_classes, kernel_size=1)
        
        self._init_bika_bias()
    
    def _init_bika_bias(self):
        for m in self.modules():
            if isinstance(m, BiKA_DWConv2d):
                nn.init.uniform_(m.bias, -2.0, 0.5)
    
    def forward(self, x):
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


__all__ = ["BiKA_DWConv2d", "BiKA_DSConvBlock", "BiKASegNet_DS"]
