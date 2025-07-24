#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void bika_linear_forward_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int I, int O
) {
    int b = blockIdx.x;
    int o = threadIdx.x;
    if (b >= B || o >= O) return;

    float acc = 0.0;
    for (int i = 0; i < I; i++) {
        float x = input[b * I + i];
        float w = weight[o * I + i];
        float b_ = bias[o * I + i];
        float prod = (x + b_) * w;
        acc += (prod >= 0) ? 1.0f : -1.0f;
    }

    output[b * O + o] = acc;
}

__global__ void bika_linear_backward_kernel(
    const float* grad_output, const float* input, const float* weight, const float* bias,
    float* grad_input, float* grad_weight, float* grad_bias,
    int B, int I, int O
) {
    int b = blockIdx.x;
    int i = threadIdx.x;
    if (b >= B || i >= I) return;

    float dx = 0.0f;
    for (int o = 0; o < O; o++) {
        float x = input[b * I + i];
        float w = weight[o * I + i];
        float b_ = bias[o * I + i];
        float z = (x + b_) * w;
        float sgrad = (z == 0.0f) ? 0.0f : 1.0f;
        float go = grad_output[b * O + o];
        dx += go * sgrad * w;
        atomicAdd(&grad_weight[o * I + i], go * sgrad * (x + b_));
        atomicAdd(&grad_bias[o * I + i], go * sgrad * w);
    }
    grad_input[b * I + i] = dx;
}

torch::Tensor bika_linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int B = input.size(0);
    int I = input.size(1);
    int O = weight.size(0);
    auto output = torch::empty({B, O}, input.options());
    bika_linear_forward_kernel<<<B, O>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, I, O
    );
    return output;
}

std::vector<torch::Tensor> bika_linear_backward(torch::Tensor grad_output,
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int B = input.size(0);
    int I = input.size(1);
    int O = weight.size(0);
    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros_like(bias);
    bika_linear_backward_kernel<<<B, I>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
        B, I, O
    );
    return {grad_input, grad_weight, grad_bias};
}
