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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bika_linear_forward",  &bika_linear_forward,  "BiKA Linear forward (CUDA)");
  m.def("bika_linear_backward", &bika_linear_backward, "BiKA Linear backward (CUDA)");

  m.def("bika_conv2d_forward",  &bika_conv2d_forward,  "BiKA Conv2d forward (stride>=1, padding>=0, dilation=1, groups=1)");
  m.def("bika_conv2d_backward", &bika_conv2d_backward, "BiKA Conv2d backward (stride>=1, padding>=0, dilation=1, groups=1)");
}
