// bika_conv2d.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_CHECK(err) TORCH_CHECK((err) == cudaSuccess, "CUDA error: ", cudaGetErrorString(err))

__global__ void bika_conv2d_forward_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    const float* __restrict__ weight,  // [O, C, K, K]
    const float* __restrict__ bias,    // [O, C, K, K]
    float* __restrict__ output,        // [B, O, Ho, Wo]
    int B, int C, int H, int W, int O, int K, int Ho, int Wo
) {
    int b = blockIdx.x;
    int o = blockIdx.y;
    if (b >= B || o >= O) return;

    for (int h_out = threadIdx.y; h_out < Ho; h_out += blockDim.y) {
        for (int w_out = threadIdx.x; w_out < Wo; w_out += blockDim.x) {
            float acc = 0.0f;
            for (int c = 0; c < C; ++c) {
                const float* w_oc = weight + ((o * C + c) * K) * K;
                const float* b_oc = bias   + ((o * C + c) * K) * K;

                for (int kh = 0; kh < K; ++kh) {
                    int h_in = h_out + kh;
                    const float* x_row = input + (((b * C + c) * H + h_in) * W);
                    const float* w_row = w_oc + kh * K;
                    const float* b_row = b_oc + kh * K;

                    for (int kw = 0; kw < K; ++kw) {
                        int w_in = w_out + kw;
                        float z = (x_row[w_in] + b_row[kw]) * w_row[kw];
                        acc += (z >= 0.0f) ? 1.0f : -1.0f;
                    }
                }
            }
            output[(((b * O + o) * Ho) + h_out) * Wo + w_out] = acc;
        }
    }
}

__global__ void bika_conv2d_backward_kernel(
    const float* __restrict__ grad_output, // [B, O, Ho, Wo]
    const float* __restrict__ input,       // [B, C, H, W]
    const float* __restrict__ weight,      // [O, C, K, K]
    const float* __restrict__ bias,        // [O, C, K, K]
    float* __restrict__ grad_input,        // [B, C, H, W]
    float* __restrict__ grad_weight,       // [O, C, K, K]
    float* __restrict__ grad_bias,         // [O, C, K, K]
    int B, int C, int H, int W, int O, int K, int Ho, int Wo
) {
    int b = blockIdx.x;
    int o = blockIdx.y;
    if (b >= B || o >= O) return;

    for (int h_out = threadIdx.y; h_out < Ho; h_out += blockDim.y) {
        for (int w_out = threadIdx.x; w_out < Wo; w_out += blockDim.x) {
            const float go = grad_output[(((b * O + o) * Ho) + h_out) * Wo + w_out];

            for (int c = 0; c < C; ++c) {
                const float* w_oc = weight + ((o * C + c) * K) * K;
                const float* b_oc = bias   + ((o * C + c) * K) * K;

                for (int kh = 0; kh < K; ++kh) {
                    int h_in = h_out + kh;

                    float* gx_row = grad_input + (((b * C + c) * H + h_in) * W);
                    const float* x_row = input + (((b * C + c) * H + h_in) * W);
                    const float* w_row = w_oc + kh * K;
                    const float* b_row = b_oc + kh * K;
                    float* gw_row = grad_weight + ((o * C + c) * K + kh) * K;
                    float* gb_row = grad_bias   + ((o * C + c) * K + kh) * K;

                    for (int kw = 0; kw < K; ++kw) {
                        int w_in = w_out + kw;

                        const float x = x_row[w_in];
                        const float w = w_row[kw];
                        const float bb = b_row[kw];
                        const float z = (x + bb) * w;

                        const float sgrad = (fabsf(z) <= 1.0f) ? 1.0f : 0.0f; // hard-tanh STE

                        atomicAdd(&gx_row[w_in], go * sgrad * w);
                        atomicAdd(&gw_row[kw],   go * sgrad * (x + bb));
                        atomicAdd(&gb_row[kw],   go * sgrad * w);
                    }
                }
            }
        }
    }
}

torch::Tensor bika_conv2d_forward(torch::Tensor input,
                                   torch::Tensor weight,
                                   torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(input.dtype()==torch::kFloat32 && weight.dtype()==torch::kFloat32 && bias.dtype()==torch::kFloat32,
                "All tensors must be float32");
    TORCH_CHECK(input.dim()==4 && weight.dim()==4 && bias.dim()==4,
                "input[B,C,H,W], weight[O,C,K,K], bias[O,C,K,K]");
    TORCH_CHECK(weight.sizes()==bias.sizes(), "weight and bias must be [O,C,K,K]");
    TORCH_CHECK(input.size(1)==weight.size(1), "C must match");
    TORCH_CHECK(weight.size(2)==weight.size(3), "K must be square");

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int O = weight.size(0), K = weight.size(2);
    TORCH_CHECK(K <= H && K <= W, "Valid conv requires K<=H and K<=W");

    int Ho = H - K + 1, Wo = W - K + 1;
    auto output = torch::empty({B, O, Ho, Wo}, input.options());

    at::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(B, O);
    dim3 block(16, 16);
    bika_conv2d_forward_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C, H, W, O, K, Ho, Wo
    );
    CUDA_CHECK(cudaGetLastError());
    return output;
}

std::vector<torch::Tensor> bika_conv2d_backward(torch::Tensor grad_output,
                                                 torch::Tensor input,
                                                 torch::Tensor weight,
                                                 torch::Tensor bias) {
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(grad_output.dtype()==torch::kFloat32 && input.dtype()==torch::kFloat32 &&
                weight.dtype()==torch::kFloat32 && bias.dtype()==torch::kFloat32,
                "All tensors must be float32");

    TORCH_CHECK(input.dim()==4 && weight.dim()==4 && bias.dim()==4 && grad_output.dim()==4,
                "input[B,C,H,W], weight[O,C,K,K], bias[O,C,K,K], grad_output[B,O,Ho,Wo]");
    TORCH_CHECK(weight.sizes()==bias.sizes(), "weight and bias must be [O,C,K,K]");
    TORCH_CHECK(input.size(1)==weight.size(1), "C must match");
    TORCH_CHECK(weight.size(2)==weight.size(3), "K must be square");

    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    grad_output = grad_output.contiguous();

    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int O = weight.size(0), K = weight.size(2);

    TORCH_CHECK(K <= H && K <= W, "Valid conv requires K<=H and K<=W");

    int Ho = H - K + 1, Wo = W - K + 1;
    TORCH_CHECK(grad_output.size(0)==B && grad_output.size(1)==O &&
                grad_output.size(2)==Ho && grad_output.size(3)==Wo,
                "grad_output must be [B,O,Ho,Wo]");

    auto grad_input  = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias   = torch::zeros_like(bias);

    at::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(B, O);
    dim3 block(16, 16);
    bika_conv2d_backward_kernel<<<grid, block, 0, stream>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
        B, C, H, W, O, K, Ho, Wo
    );
    CUDA_CHECK(cudaGetLastError());

    return {grad_input, grad_weight, grad_bias};
}

