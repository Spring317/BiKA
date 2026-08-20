// bika_binding.cpp
#include <torch/extension.h>

// linear (unchanged)
torch::Tensor bika_linear_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias);
std::vector<torch::Tensor> bika_linear_backward(torch::Tensor grad_output,
                                                torch::Tensor input,
                                                torch::Tensor weight,
                                                torch::Tensor bias);

// conv (UPDATED: padding + stride)
torch::Tensor bika_conv2d_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  torch::Tensor out_scale,
                                  torch::Tensor out_shift,
                                  torch::Tensor packed_weight,
                                  torch::Tensor neg_bias_half,
                                  bool do_relu,
                                  int pad_h,
                                  int pad_w,
                                  int stride_h,
                                  int stride_w);
std::vector<torch::Tensor> bika_conv2d_backward(torch::Tensor grad_output,
                                                torch::Tensor input,
                                                torch::Tensor weight,
                                                torch::Tensor bias,
                                                int pad_h,
                                                int pad_w,
                                                int stride_h,
                                                int stride_w);

torch::Tensor bika_conv2d_forward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// V2 optimized CPU kernel (pre-binarize + 4-channel tiling)
torch::Tensor bika_conv2d_forward_cpu_v2_dispatch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// V3 optimized CPU kernel (im2col + quad-channel fused XNOR)
torch::Tensor bika_conv2d_forward_cpu_v3_dispatch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// V4 optimized CPU kernel (8-channel straight-line AVX2)
torch::Tensor bika_conv2d_forward_cpu_v4(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// V5 optimized CPU kernel (Channel-Stationary L1-Resident AVX2)
torch::Tensor bika_conv2d_forward_cpu_v5(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// Strategy 3+4: Int8 AVX2 SIMD CPU kernel
torch::Tensor bika_conv2d_forward_cpu_int8(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// Strategy 3+4 (V2): 2D-Tiled Int8 AVX2 SIMD CPU kernel with pre-quantized bias
torch::Tensor bika_conv2d_forward_cpu_int8_v2(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    torch::Tensor packed_bias_i8,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// Strategy 3+4 (V3): 4-Pixel x 8-Channel Int8 AVX2 SIMD CPU kernel
torch::Tensor bika_conv2d_forward_cpu_int8_v3(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    torch::Tensor packed_bias_i8,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

// Strategy 4 (Advanced): Channels-Last (NHWC) Contiguous Streaming Int8 AVX2 CPU kernel
torch::Tensor bika_conv2d_forward_cpu_nhwc(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    torch::Tensor packed_bias_i8,
    bool do_relu,
    int pad_h, int pad_w,
    int stride_h, int stride_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bika_linear_forward", &bika_linear_forward, "BiKA Linear Forward (CUDA)");
    m.def("bika_linear_backward", &bika_linear_backward, "BiKA Linear Backward (CUDA)");
    m.def("bika_conv2d_forward", &bika_conv2d_forward, "BiKA Conv2d Forward (CUDA)");
    m.def("bika_conv2d_backward", &bika_conv2d_backward, "BiKA Conv2d Backward (CUDA)");
    m.def("bika_conv2d_forward_cpu", &bika_conv2d_forward_cpu, "BiKA Conv2d Forward (CPU V1)");
    m.def("bika_conv2d_forward_cpu_v2", &bika_conv2d_forward_cpu_v2_dispatch, "BiKA Conv2d Forward (CPU V2)");
    m.def("bika_conv2d_forward_cpu_v3", &bika_conv2d_forward_cpu_v3_dispatch, "BiKA Conv2d Forward (CPU V3 - im2col + quad)");
    m.def("bika_conv2d_forward_cpu_v4", &bika_conv2d_forward_cpu_v4, "BiKA Conv2d Forward (CPU V4 - 8ch straight-line AVX2)");
    m.def("bika_conv2d_forward_cpu_v5", &bika_conv2d_forward_cpu_v5, "BiKA Conv2d Forward (CPU V5 - Channel-Stationary AVX2)");
    m.def("bika_conv2d_forward_cpu_int8", &bika_conv2d_forward_cpu_int8, "BiKA Conv2d Forward (CPU Int8 AVX2 SIMD)");
    m.def("bika_conv2d_forward_cpu_int8_v2", &bika_conv2d_forward_cpu_int8_v2, "BiKA Conv2d Forward (CPU Int8 V2 2D-Tiled AVX2 SIMD)");
    m.def("bika_conv2d_forward_cpu_int8_v3", &bika_conv2d_forward_cpu_int8_v3, "BiKA Conv2d Forward (CPU Int8 V3 4P8CH AVX2 SIMD)");
    m.def("bika_conv2d_forward_cpu_nhwc", &bika_conv2d_forward_cpu_nhwc, "BiKA Conv2d Forward (CPU Int8 NHWC Streaming AVX2 SIMD)");
}
