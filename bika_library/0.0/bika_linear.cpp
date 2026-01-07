#include <torch/extension.h>

torch::Tensor bika_linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
std::vector<torch::Tensor> bika_linear_backward(torch::Tensor grad_out, torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bika_linear_forward", &bika_linear_forward, "BiKA Linear forward (CUDA)");
    m.def("bika_linear_backward", &bika_linear_backward, "BiKA Linear backward (CUDA)");
}
