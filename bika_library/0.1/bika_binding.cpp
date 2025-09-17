#include <torch/extension.h>

// Declarations (implemented in the .cu files)
torch::Tensor bika_linear_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias);
std::vector<torch::Tensor> bika_linear_backward(torch::Tensor grad_output,
                                                torch::Tensor input,
                                                torch::Tensor weight,
                                                torch::Tensor bias);

torch::Tensor bika_conv2d_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias);
std::vector<torch::Tensor> bika_conv2d_backward(torch::Tensor grad_output,
                                                torch::Tensor input,
                                                torch::Tensor weight,
                                                torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bika_linear_forward",  &bika_linear_forward,  "BiKA Linear forward (CUDA)");
  m.def("bika_linear_backward", &bika_linear_backward, "BiKA Linear backward (CUDA)");
  m.def("bika_conv2d_forward",  &bika_conv2d_forward,  "BiKA Conv2d forward (CUDA)");
  m.def("bika_conv2d_backward", &bika_conv2d_backward, "BiKA Conv2d backward (CUDA)");
}

