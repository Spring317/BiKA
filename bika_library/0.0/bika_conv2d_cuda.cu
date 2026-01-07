#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void bika_conv2d_forward_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C, int H, int W, int O, int K, int Ho, int Wo
) {
    int b = blockIdx.x;
    int o = blockIdx.y;
    int h_out = threadIdx.y;
    int w_out = threadIdx.x;
    if (b >= B || o >= O || h_out >= Ho || w_out >= Wo) return;

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                if (h_in < H && w_in < W) {
                    int idx = ((b * C + c) * H + h_in) * W + w_in;
                    int idx_w = ((o * C + c) * K + kh) * K + kw;
                    float val = input[idx];
                    float w = weight[idx_w];
                    float b_ = bias[idx_w];
                    float z = (val + b_) * w;
                    acc += (z >= 0.0f) ? 1.0f : -1.0f;
                }
            }
        }
    }
    output[((b * O + o) * Ho + h_out) * Wo + w_out] = acc;
}

__global__ void bika_conv2d_backward_kernel(
    const float* grad_output, const float* input, const float* weight, const float* bias,
    float* grad_input, float* grad_weight, float* grad_bias,
    int B, int C, int H, int W, int O, int K, int Ho, int Wo
) {
    int b = blockIdx.x;
    int o = blockIdx.y;
    int h_out = threadIdx.y;
    int w_out = threadIdx.x;
    if (b >= B || o >= O || h_out >= Ho || w_out >= Wo) return;

    float go = grad_output[((b * O + o) * Ho + h_out) * Wo + w_out];

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                if (h_in >= H || w_in >= W) continue;

                int idx_x = ((b * C + c) * H + h_in) * W + w_in;
                int idx_w = ((o * C + c) * K + kh) * K + kw;
                float x_val = input[idx_x];
                float w_val = weight[idx_w];
                float b_val = bias[idx_w];
                float z = (x_val + b_val) * w_val;
                float sgrad = (z == 0.0f) ? 0.0f : 1.0f;

                atomicAdd(&grad_input[idx_x], go * sgrad * w_val);
                atomicAdd(&grad_weight[idx_w], go * sgrad * (x_val + b_val));
                atomicAdd(&grad_bias[idx_w], go * sgrad * w_val);
            }
        }
    }
}

torch::Tensor bika_conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int O = weight.size(0), K = weight.size(2);
    int Ho = H - K + 1, Wo = W - K + 1;
    auto output = torch::empty({B, O, Ho, Wo}, input.options());

    dim3 blocks(B, O);
    dim3 threads(Wo, Ho);

    bika_conv2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C, H, W, O, K, Ho, Wo
    );

    return output;
}

std::vector<torch::Tensor> bika_conv2d_backward(torch::Tensor grad_output,
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int O = weight.size(0), K = weight.size(2);
    int Ho = H - K + 1, Wo = W - K + 1;

    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros_like(bias);

    dim3 blocks(B, O);
    dim3 threads(Wo, Ho);

    bika_conv2d_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
        B, C, H, W, O, K, Ho, Wo
    );

    return {grad_input, grad_weight, grad_bias};
}
