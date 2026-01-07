#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>  // fabsf

// ===================================================
// FORWARD KERNEL (same as before)
// ===================================================
__global__ void bika_conv2d_forward_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    const float* __restrict__ weight,  // [O, C, K, K]
    const float* __restrict__ bias,    // [O, C, K, K]
    float* __restrict__ output,        // [B, O, Ho, Wo]
    int B, int C, int H, int W, int O, int K, int Ho, int Wo,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    int b = blockIdx.x;
    int o = blockIdx.y;
    if (b >= B || o >= O) return;

    for (int h_out = threadIdx.y; h_out < Ho; h_out += blockDim.y) {
        for (int w_out = threadIdx.x; w_out < Wo; w_out += blockDim.x) {
            float acc = 0.0f;

            const int h_base = h_out * stride_h - pad_h;
            const int w_base = w_out * stride_w - pad_w;

            for (int c = 0; c < C; ++c) {
                const float* w_oc = weight + ((o * C + c) * K) * K;
                const float* b_oc = bias   + ((o * C + c) * K) * K;

                for (int kh = 0; kh < K; ++kh) {
                    const int h_in = h_base + kh;
                    const float* w_row = w_oc + kh * K;
                    const float* b_row = b_oc + kh * K;

                    const bool row_in = (h_in >= 0 && h_in < H);
                    const float* x_row = row_in ? (input + (((b * C + c) * H + h_in) * W)) : nullptr;

                    for (int kw = 0; kw < K; ++kw) {
                        const int w_in = w_base + kw;
                        const bool inb = row_in && (w_in >= 0 && w_in < W);

                        const float x  = inb ? x_row[w_in] : 0.0f; // zero padding
                        const float w  = w_row[kw];
                        const float bb = b_row[kw];
                        const float z  = (x + bb) * w;

                        acc += (z >= 0.0f) ? 1.0f : -1.0f;
                    }
                }
            }
            output[(((b * O + o) * Ho) + h_out) * Wo + w_out] = acc; // no sign after sum
        }
    }
}

// ===================================================
// BACKWARD KERNELS (no atomics)
// ===================================================

// 1) grad_weight & grad_bias kernel (each thread computes one (o,c,kh,kw))
__global__ void bika_conv2d_backward_wb_kernel(
    const float* __restrict__ grad_output, // [B, O, Ho, Wo]
    const float* __restrict__ input,       // [B, C, H, W]
    const float* __restrict__ weight,      // [O, C, K, K] (read-only, for z)
    const float* __restrict__ bias,        // [O, C, K, K] (read-only, for z)
    float* __restrict__ grad_weight,       // [O, C, K, K]
    float* __restrict__ grad_bias,         // [O, C, K, K]
    int B, int C, int H, int W, int O, int K, int Ho, int Wo,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    int o  = blockIdx.x;                 // one output channel per blockIdx.x
    int c  = blockIdx.y;                 // one input channel per blockIdx.y
    int tid = threadIdx.x;               // use 1D threads over kernel positions
    if (o >= O || c >= C) return;

    // map thread to (kh, kw)
    const int KK = K * K;
    for (int idx = tid; idx < KK; idx += blockDim.x) {
        int kh = idx / K;
        int kw = idx % K;

        const float w  = weight[((o * C + c) * K + kh) * K + kw];
        const float bb = bias  [((o * C + c) * K + kh) * K + kw];

        float acc_w = 0.0f;  // grad w[o,c,kh,kw]
        float acc_b = 0.0f;  // grad b[o,c,kh,kw]

        // loop over batch and all output spatial positions
        for (int b = 0; b < B; ++b) {
            for (int h_out = 0; h_out < Ho; ++h_out) {
                const int h_in = h_out * stride_h - pad_h + kh;
                const bool row_in = (h_in >= 0 && h_in < H);
                const float* x_row = row_in ? (input + (((b * C + c) * H + h_in) * W)) : nullptr;

                for (int w_out = 0; w_out < Wo; ++w_out) {
                    const int w_in = w_out * stride_w - pad_w + kw;
                    const bool inb = row_in && (w_in >= 0 && w_in < W);

                    const float x  = inb ? x_row[w_in] : 0.0f;      // zero padding
                    const float z  = (x + bb) * w;
                    const float sgrad = (fabsf(z) <= 1.0f) ? 1.0f : 0.0f;

                    const float go = grad_output[(((b * O + o) * Ho) + h_out) * Wo + w_out];

                    acc_w += go * sgrad * (x + bb); // dL/dw
                    acc_b += go * sgrad * w;        // dL/db
                }
            }
        }

        // store results (no atomics needed; unique writer)
        grad_weight[((o * C + c) * K + kh) * K + kw] = acc_w;
        grad_bias  [((o * C + c) * K + kh) * K + kw] = acc_b;
    }
}

// 2) grad_input kernel (each thread computes one (b,c,h_in,w_in))
__global__ void bika_conv2d_backward_input_kernel(
    const float* __restrict__ grad_output, // [B, O, Ho, Wo]
    const float* __restrict__ input,       // [B, C, H, W] (read for z; could be dropped if you accept x-only via bounds)
    const float* __restrict__ weight,      // [O, C, K, K]
    const float* __restrict__ bias,        // [O, C, K, K]
    float* __restrict__ grad_input,        // [B, C, H, W]
    int B, int C, int H, int W, int O, int K, int Ho, int Wo,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    int b = blockIdx.x;  // batch
    int c = blockIdx.y;  // channel
    if (b >= B || c >= C) return;

    // tile (h,w) by threads
    for (int h_in = threadIdx.y; h_in < H; h_in += blockDim.y) {
        for (int w_in = threadIdx.x; w_in < W; w_in += blockDim.x) {
            float acc = 0.0f;

            // which output (h_out,w_out) use this (h_in,w_in)?
            // h_in = h_out*stride_h - pad_h + kh  =>  h_out = (h_in + pad_h - kh)/stride_h
            // w_in = w_out*stride_w - pad_w + kw  =>  w_out = (w_in + pad_w - kw)/stride_w

            for (int o = 0; o < O; ++o) {
                const float* w_oc = weight + ((o * C + c) * K) * K;
                const float* b_oc = bias   + ((o * C + c) * K) * K;

                for (int kh = 0; kh < K; ++kh) {
                    const int num_h = (h_in + pad_h - kh);
                    if (num_h % stride_h != 0) continue;  // no matching output row
                    const int h_out = num_h / stride_h;
                    if (h_out < 0 || h_out >= Ho) continue;

                    const float* w_row = w_oc + kh * K;
                    const float* b_row = b_oc + kh * K;

                    for (int kw = 0; kw < K; ++kw) {
                        const int num_w = (w_in + pad_w - kw);
                        if (num_w % stride_w != 0) continue;
                        const int w_out = num_w / stride_w;
                        if (w_out < 0 || w_out >= Wo) continue;

                        const float w  = w_row[kw];
                        const float bb = b_row[kw];

                        // x at (h_in,w_in) is in-bounds here by construction
                        const float x  = input[(((b * C + c) * H + h_in) * W) + w_in];
                        const float z  = (x + bb) * w;
                        const float sgrad = (fabsf(z) <= 1.0f) ? 1.0f : 0.0f;

                        const float go = grad_output[(((b * O + o) * Ho) + h_out) * Wo + w_out];

                        acc += go * sgrad * w;  // dL/dx
                    }
                }
            }

            grad_input[(((b * C + c) * H + h_in) * W) + w_in] = acc;
        }
    }
}

// ===================================================
// LAUNCHERS
// ===================================================

torch::Tensor bika_conv2d_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int pad_h,
                                  int pad_w,
                                  int stride_h,
                                  int stride_w) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(input.dtype()==torch::kFloat32 && weight.dtype()==torch::kFloat32 && bias.dtype()==torch::kFloat32,
                "All tensors must be float32");
    TORCH_CHECK(input.dim()==4 && weight.dim()==4 && bias.dim()==4,
                "input[B,C,H,W], weight[O,C,K,K], bias[O,C,K,K]");
    TORCH_CHECK(weight.sizes()==bias.sizes(), "weight and bias must be [O,C,K,K]");
    TORCH_CHECK(input.size(1)==weight.size(1), "C must match");
    TORCH_CHECK(weight.size(2)==weight.size(3), "K must be square");
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "padding must be >= 0");
    TORCH_CHECK(stride_h >= 1 && stride_w >= 1, "stride must be >= 1");

    input  = input.contiguous();
    weight = weight.contiguous();
    bias   = bias.contiguous();

    const int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int O = weight.size(0), K = weight.size(2);

    const int Ho = (H + 2*pad_h - K) / stride_h + 1;
    const int Wo = (W + 2*pad_w - K) / stride_w + 1;
    TORCH_CHECK(Ho > 0 && Wo > 0, "Invalid output size: check K, stride, padding");

    auto output = torch::empty({B, O, Ho, Wo}, input.options());

    const int dev = input.get_device();
    TORCH_CHECK(dev >= 0, "Input must be a CUDA tensor");
    cudaSetDevice(dev);

    dim3 gridF(B, O);
    dim3 blockF(16, 16);
    bika_conv2d_forward_kernel<<<gridF, blockF>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C, H, W, O, K, Ho, Wo, pad_h, pad_w, stride_h, stride_w
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA launch failed (bika_conv2d_forward).");
    return output;
}

std::vector<torch::Tensor> bika_conv2d_backward(torch::Tensor grad_output,
                                                 torch::Tensor input,
                                                 torch::Tensor weight,
                                                 torch::Tensor bias,
                                                 int pad_h,
                                                 int pad_w,
                                                 int stride_h,
                                                 int stride_w) {
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
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "padding must be >= 0");
    TORCH_CHECK(stride_h >= 1 && stride_w >= 1, "stride must be >= 1");

    input        = input.contiguous();
    weight       = weight.contiguous();
    bias         = bias.contiguous();
    grad_output  = grad_output.contiguous();

    const int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int O = weight.size(0), K = weight.size(2);

    const int Ho = (H + 2*pad_h - K) / stride_h + 1;
    const int Wo = (W + 2*pad_w - K) / stride_w + 1;
    TORCH_CHECK(Ho > 0 && Wo > 0, "Invalid output size: check K, stride, padding");
    TORCH_CHECK(grad_output.size(0)==B && grad_output.size(1)==O &&
                grad_output.size(2)==Ho && grad_output.size(3)==Wo,
                "grad_output must be [B,O,Ho,Wo]");

    auto grad_input  = torch::empty_like(input);  // we'll fully write it
    auto grad_weight = torch::empty_like(weight); // fully written
    auto grad_bias   = torch::empty_like(bias);   // fully written

    const int dev = input.get_device();
    TORCH_CHECK(dev >= 0, "Input must be a CUDA tensor");
    cudaSetDevice(dev);

    // --- launch grad_weight/bias kernel ---
    dim3 gridWB(O, C);
    int threadsWB = (K * K <= 256) ? (K * K) : 256;  // one thread per (kh,kw), grid-stride if needed
    bika_conv2d_backward_wb_kernel<<<gridWB, threadsWB>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
        B, C, H, W, O, K, Ho, Wo, pad_h, pad_w, stride_h, stride_w
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA launch failed (bika_conv2d_backward_wb).");

    // --- launch grad_input kernel ---
    dim3 gridGI(B, C);
    dim3 blockGI(16, 16);
    bika_conv2d_backward_input_kernel<<<gridGI, blockGI>>>(
        grad_output.data_ptr<float>(), input.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        B, C, H, W, O, K, Ho, Wo, pad_h, pad_w, stride_h, stride_w
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA launch failed (bika_conv2d_backward_input).");

    return {grad_input, grad_weight, grad_bias};
}
