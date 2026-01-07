// bika_linear.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>  // fabsf

// -------------------- KERNELS --------------------
__global__ void bika_linear_forward_kernel(
    const float* __restrict__ input,   // [B, I]
    const float* __restrict__ weight,  // [O, I]
    const float* __restrict__ bias,    // [O, I]
    float* __restrict__ output,        // [B, O]
    int B, int I, int O
) {
    int b = blockIdx.x;
    if (b >= B) return;

    for (int o = threadIdx.x; o < O; o += blockDim.x) {
        float acc = 0.0f;
        const float* wrow = weight + o * I;
        const float* brow = bias   + o * I;
        const float* xin  = input  + b * I;

        for (int i = 0; i < I; ++i) {
            float z = (xin[i] + brow[i]) * wrow[i];
            acc += (z >= 0.0f) ? 1.0f : -1.0f;
        }
        output[b * O + o] = acc;  // no sign after the sum
    }
}

__global__ void bika_linear_backward_kernel(
    const float* __restrict__ grad_output, // [B, O]
    const float* __restrict__ input,       // [B, I]
    const float* __restrict__ weight,      // [O, I]
    const float* __restrict__ bias,        // [O, I]
    float* __restrict__ grad_input,        // [B, I]
    float* __restrict__ grad_weight,       // [O, I]
    float* __restrict__ grad_bias,         // [O, I]
    int B, int I, int O
) {
    int b = blockIdx.x;
    if (b >= B) return;

    for (int i = threadIdx.x; i < I; i += blockDim.x) {
        float dx = 0.0f;
        const float xbi = input[b * I + i];

        for (int o = 0; o < O; ++o) {
            const float woi = weight[o * I + i];
            const float boi = bias[o * I + i];
            const float z   = (xbi + boi) * woi;

            // Hard-tanh STE for sign: pass grad when |z| <= 1
            const float sgrad = (fabsf(z) <= 1.0f) ? 1.0f : 0.0f;
            const float go = grad_output[b * O + o];

            dx += go * sgrad * woi;                         // dL/dx
            atomicAdd(&grad_weight[o * I + i], go * sgrad * (xbi + boi)); // dL/dw
            atomicAdd(&grad_bias[o * I + i],   go * sgrad * woi);         // dL/db
        }
        grad_input[b * I + i] = dx;
    }
}

// -------------------- LAUNCHERS --------------------
torch::Tensor bika_linear_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(input.dtype()==torch::kFloat32 && weight.dtype()==torch::kFloat32 && bias.dtype()==torch::kFloat32,
                "All tensors must be float32");
    TORCH_CHECK(input.dim()==2 && weight.dim()==2 && bias.dim()==2, "input[B,I], weight[O,I], bias[O,I]");
    TORCH_CHECK(weight.sizes()==bias.sizes(), "weight and bias must have same shape [O,I]");
    TORCH_CHECK(input.size(1)==weight.size(1), "I must match between input and weight");

    input  = input.contiguous();
    weight = weight.contiguous();
    bias   = bias.contiguous();

    const int B = input.size(0);
    const int I = input.size(1);
    const int O = weight.size(0);

    auto output = torch::empty({B, O}, input.options());

    // Set device and launch on default stream
    int dev = input.get_device();
    TORCH_CHECK(dev >= 0, "Input must be a CUDA tensor");
    cudaSetDevice(dev);

    const int block = 256;
    const dim3 grid(B);

    bika_linear_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, I, O
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA launch failed (bika_linear_forward).");
    return output;
}

std::vector<torch::Tensor> bika_linear_backward(torch::Tensor grad_output,
                                                torch::Tensor input,
                                                torch::Tensor weight,
                                                torch::Tensor bias) {
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(grad_output.dtype()==torch::kFloat32 && input.dtype()==torch::kFloat32 &&
                weight.dtype()==torch::kFloat32 && bias.dtype()==torch::kFloat32,
                "All tensors must be float32");
    TORCH_CHECK(input.dim()==2 && weight.dim()==2 && bias.dim()==2 && grad_output.dim()==2,
                "input[B,I], weight[O,I], bias[O,I], grad_output[B,O]");
    TORCH_CHECK(weight.sizes()==bias.sizes(), "weight and bias must have same shape [O,I]");
    TORCH_CHECK(input.size(1)==weight.size(1), "I must match between input and weight");
    TORCH_CHECK(grad_output.size(0)==input.size(0) && grad_output.size(1)==weight.size(0),
                "grad_output must be [B,O]");

    grad_output = grad_output.contiguous();
    input  = input.contiguous();
    weight = weight.contiguous();
    bias   = bias.contiguous();

    const int B = input.size(0);
    const int I = input.size(1);
    const int O = weight.size(0);

    auto grad_input  = torch::empty_like(input);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias   = torch::zeros_like(bias);

    int dev = input.get_device();
    TORCH_CHECK(dev >= 0, "Input must be a CUDA tensor");
    cudaSetDevice(dev);

    const int block = 256;
    const dim3 grid(B);

    bika_linear_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
        B, I, O
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA launch failed (bika_linear_backward).");

    return {grad_input, grad_weight, grad_bias};
}
