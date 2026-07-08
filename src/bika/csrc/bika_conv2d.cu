#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>  // fabsf

// ===================================================
// OPTIMIZED FORWARD KERNEL V4: O-Tiling (Register Reuse)
//
// By computing TILE_O output channels per thread, we reduce the number
// of global memory reads of the input feature map `x` by TILE_O times.
// We use TILE_O = 4 to keep shared memory under 48KB even for the largest layers.
// ===================================================

#define FWD_BLOCK_SIZE 256
#define TILE_O 4
#define TILE_W 4

template <int K_TMPL>
__global__ void bika_conv2d_forward_kernel(
    const float* __restrict__ input,   // [B, C, H, W]
    const float* __restrict__ weight,  // [O, C, K, K]
    const float* __restrict__ bias,    // [O, C, K, K]
    float* __restrict__ output,        // [B, O, Ho, Wo]
    const float* __restrict__ out_scale, // [O]
    const float* __restrict__ out_shift, // [O]
    const int* __restrict__ packed_weight, // [O, num_words]
    const __half* __restrict__ neg_bias_half, // [O, C, K, K] pre-negated half-precision bias
    bool do_relu,
    int B, int C, int H, int W, int O, int K, int Ho, int Wo,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    const int num_spatial_tiles_w = (Wo + TILE_W - 1) / TILE_W;
    const int num_spatial_tiles_total = Ho * num_spatial_tiles_w;
    
    const int b = blockIdx.z;
    const int o_base = blockIdx.y * TILE_O;
    const int pixel_idx = blockIdx.x * FWD_BLOCK_SIZE + threadIdx.x;

    if (b >= B || o_base >= O || pixel_idx >= num_spatial_tiles_total) return;

    const int ckk = C * K_TMPL * K_TMPL;
    const int num_words = (ckk + 31) / 32;

    extern __shared__ unsigned int smem_fwd[];
    unsigned int* smem_w_packed = smem_fwd;
    // Use half-precision for bias in shared memory to halve bandwidth
    __half* smem_bias_h = (__half*)(smem_w_packed + TILE_O * num_words);

    for (int to = 0; to < TILE_O; ++to) {
        const int o = o_base + to;
        if (o >= O) continue;

        const int w_offset = o * ckk;

        for (int i = threadIdx.x; i < ckk; i += FWD_BLOCK_SIZE) {
            if (neg_bias_half != nullptr) {
                smem_bias_h[to * ckk + i] = neg_bias_half[w_offset + i];
            } else {
                smem_bias_h[to * ckk + i] = __float2half(__ldg(&bias[w_offset + i]));
            }
        }

        for (int word = threadIdx.x; word < num_words; word += FWD_BLOCK_SIZE) {
            if (packed_weight != nullptr) {
            smem_w_packed[to * num_words + word] = packed_weight[(o_base + to) * num_words + word];
        } else {
            unsigned int packed = 0u;
            const int start = word * 32;
            const int end = min(start + 32, ckk);
            for (int k = 0; k < (end - start); ++k) {
                if (__ldg(&weight[w_offset + start + k]) >= 0.0f) {
                    packed |= (1u << k);
                }
            }
            smem_w_packed[to * num_words + word] = packed;
        }
        }
    }
    
    __syncthreads();

    const int h_out = pixel_idx / num_spatial_tiles_w;
    const int w_out_base = (pixel_idx % num_spatial_tiles_w) * TILE_W;
    const int h_base = h_out * stride_h - pad_h;

    int acc[TILE_W][TILE_O] = {0};
    unsigned int x_bits[TILE_W][TILE_O] = {0u};
    
    int bit = 0;
    int word_idx = 0;
    int elem = 0;

    for (int c = 0; c < C; ++c) {
        const int input_ch_base = (b * C + c) * H;

        #pragma unroll
        for (int kh = 0; kh < K_TMPL; ++kh) {
            const int h_in = h_base + kh;
            const bool row_valid = (h_in >= 0 && h_in < H);
            const int row_offset = row_valid ? (input_ch_base + h_in) * W : 0;

            #pragma unroll
            for (int kw = 0; kw < K_TMPL; ++kw) {
                
                float x_arr[TILE_W];
                #pragma unroll
                for (int tw = 0; tw < TILE_W; ++tw) {
                    const int w_out = w_out_base + tw;
                    const int w_in = w_out * stride_w - pad_w + kw;
                    const bool valid = row_valid && (w_out < Wo) && (w_in >= 0 && w_in < W);
                    x_arr[tw] = valid ? __ldg(&input[row_offset + w_in]) : 0.0f;
                }

                #pragma unroll
                for (int to = 0; to < TILE_O; ++to) {
                    if (o_base + to < O) {
                        const __half sb = smem_bias_h[to * ckk + elem];
                        
                        #pragma unroll
                        for (int tw = 0; tw < TILE_W; ++tw) {
                            if (__hge(__float2half(x_arr[tw]), sb)) {
                                x_bits[tw][to] |= (1u << bit);
                            }
                        }
                    }
                }

                ++bit;
                if (bit == 32) {
                    #pragma unroll
                    for (int to = 0; to < TILE_O; ++to) {
                        if (o_base + to < O) {
                            unsigned int w_bits = smem_w_packed[to * num_words + word_idx];
                            
                            #pragma unroll
                            for (int tw = 0; tw < TILE_W; ++tw) {
                                const unsigned int xnor_val = ~(x_bits[tw][to] ^ w_bits);
                                acc[tw][to] += (2 * __popc(xnor_val) - 32);
                                x_bits[tw][to] = 0u;
                            }
                        }
                    }
                    bit = 0;
                    word_idx++;
                }
                elem++;
            }
        }
    }

    if (bit > 0) {
        const unsigned int mask = (1u << bit) - 1u;
        #pragma unroll
        for (int to = 0; to < TILE_O; ++to) {
            if (o_base + to < O) {
                const unsigned int w_bits = smem_w_packed[to * num_words + word_idx];
                
                #pragma unroll
                for (int tw = 0; tw < TILE_W; ++tw) {
                    const unsigned int xnor_val = (~(x_bits[tw][to] ^ w_bits)) & mask;
                    acc[tw][to] += (2 * __popc(xnor_val) - bit);
                }
            }
        }
    }

    #pragma unroll
    for (int tw = 0; tw < TILE_W; ++tw) {
        const int w_out = w_out_base + tw;
        if (w_out < Wo) {
            #pragma unroll
            for (int to = 0; to < TILE_O; ++to) {
                const int o = o_base + to;
                if (o < O) {
                    float val = (float)acc[tw][to];
                    if (out_scale != nullptr) {
                        val = val * __ldg(&out_scale[o]) + __ldg(&out_shift[o]);
                    }
                    if (do_relu && val < 0.0f) {
                        val = 0.0f;
                    }
                    output[((b * O + o) * Ho + h_out) * Wo + w_out] = val;
                }
            }
        }
    }
}

// ===================================================
// BACKWARD KERNELS (unchanged)
// ===================================================

__global__ void bika_conv2d_backward_wb_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int B, int C, int H, int W, int O, int K, int Ho, int Wo,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    int o = blockIdx.x;
    int c = blockIdx.y;

    if (o >= O || c >= C) return;

    int tid = threadIdx.x;
    const int KK = K * K;

    float acc_w[9] = {0.0f};
    float acc_b[9] = {0.0f};
    float w_reg[9] = {0.0f};
    float bb_reg[9] = {0.0f};

    for (int kidx = 0; kidx < KK; ++kidx) {
        int param_idx = ((o * C + c) * K * K) + kidx;
        w_reg[kidx] = weight[param_idx];
        bb_reg[kidx] = bias[param_idx];
    }

    const long long N = static_cast<long long>(B) * Ho * Wo;

    for (long long n = tid; n < N; n += blockDim.x) {
        int w_out = static_cast<int>(n % Wo);
        long long t = n / Wo;
        int h_out = static_cast<int>(t % Ho);
        int b = static_cast<int>(t / Ho);

        const float go = grad_output[(((b * O + o) * Ho + h_out) * Wo) + w_out];

        for (int kidx = 0; kidx < KK; ++kidx) {
            int kh = kidx / K;
            int kw = kidx % K;

            const int h_in = h_out * stride_h - pad_h + kh;
            const int w_in = w_out * stride_w - pad_w + kw;

            const bool inb = (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W);
            const float x = inb ? input[(((b * C + c) * H + h_in) * W) + w_in] : 0.0f;

            const float z = (x + bb_reg[kidx]) * w_reg[kidx];
            const float sgrad = (fabsf(z) <= 1.0f) ? 1.0f : 0.0f;

            acc_w[kidx] += go * sgrad * (x + bb_reg[kidx]);
            acc_b[kidx] += go * sgrad * w_reg[kidx];
        }
    }

    extern __shared__ float smem[]; 
    
    for (int kidx = 0; kidx < KK; ++kidx) {
        smem[kidx * blockDim.x + tid] = acc_w[kidx];
        smem[(KK + kidx) * blockDim.x + tid] = acc_b[kidx];
    }
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            for (int kidx = 0; kidx < KK; ++kidx) {
                smem[kidx * blockDim.x + tid] += smem[kidx * blockDim.x + tid + offset];
                smem[(KK + kidx) * blockDim.x + tid] += smem[(KK + kidx) * blockDim.x + tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int kidx = 0; kidx < KK; ++kidx) {
            int param_idx = ((o * C + c) * K * K) + kidx;
            grad_weight[param_idx] = smem[kidx * blockDim.x + 0];
            grad_bias[param_idx] = smem[(KK + kidx) * blockDim.x + 0];
        }
    }
}

__global__ void bika_conv2d_backward_input_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ grad_input,
    int B, int C, int H, int W, int O, int K, int Ho, int Wo,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= B || c >= C) return;

    for (int h_in = threadIdx.y; h_in < H; h_in += blockDim.y) {
        for (int w_in = threadIdx.x; w_in < W; w_in += blockDim.x) {
            float acc = 0.0f;

            for (int o = 0; o < O; ++o) {
                const float* w_oc = weight + ((o * C + c) * K) * K;
                const float* b_oc = bias   + ((o * C + c) * K) * K;

                for (int kh = 0; kh < K; ++kh) {
                    const int num_h = (h_in + pad_h - kh);
                    if (num_h % stride_h != 0) continue;
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

                        const float x  = input[(((b * C + c) * H + h_in) * W) + w_in];
                        const float z  = (x + bb) * w;
                        const float sgrad = (fabsf(z) <= 1.0f) ? 1.0f : 0.0f;

                        const float go = grad_output[(((b * O + o) * Ho) + h_out) * Wo + w_out];

                        acc += go * sgrad * w;
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
                                  torch::Tensor out_scale,
                                  torch::Tensor out_shift,
                                  torch::Tensor packed_weight,
                                  torch::Tensor neg_bias_half,
                                  bool do_relu,
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

    const int num_spatial_tiles_w = (Wo + TILE_W - 1) / TILE_W;
    const int num_spatial_tiles_total = Ho * num_spatial_tiles_w;
    const int num_spatial_blocks = (num_spatial_tiles_total + FWD_BLOCK_SIZE - 1) / FWD_BLOCK_SIZE;
    
    // Grid Y is now O / TILE_O
    const int blocks_o = (O + TILE_O - 1) / TILE_O;

    dim3 grid(num_spatial_blocks, blocks_o, B);
    dim3 block(FWD_BLOCK_SIZE);

    const int ckk = C * K * K;
    const int num_words = (ckk + 31) / 32;
    size_t shared_bytes = TILE_O * num_words * sizeof(unsigned int) + TILE_O * ckk * sizeof(__half);

    const float* scale_ptr = nullptr;
    const float* shift_ptr = nullptr;
    const int* pweight_ptr = nullptr;
    if (out_scale.numel() > 0 && out_shift.numel() > 0) {
        out_scale = out_scale.contiguous();
        out_shift = out_shift.contiguous();
        scale_ptr = out_scale.data_ptr<float>();
        shift_ptr = out_shift.data_ptr<float>();
    }
    if (packed_weight.numel() > 0) {
        TORCH_CHECK(packed_weight.is_cuda() && packed_weight.dtype() == torch::kInt32, "packed_weight must be CUDA int32");
        packed_weight = packed_weight.contiguous();
        pweight_ptr = packed_weight.data_ptr<int>();
    }
    const __half* neg_bias_half_ptr = nullptr;
    if (neg_bias_half.numel() > 0) {
        TORCH_CHECK(neg_bias_half.is_cuda() && neg_bias_half.dtype() == torch::kFloat16, "neg_bias_half must be CUDA float16");
        neg_bias_half = neg_bias_half.contiguous();
        neg_bias_half_ptr = reinterpret_cast<const __half*>(neg_bias_half.data_ptr<at::Half>());
    }

    if (K == 1) {
        bika_conv2d_forward_kernel<1><<<grid, block, shared_bytes>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            scale_ptr, shift_ptr, pweight_ptr, neg_bias_half_ptr, do_relu,
            B, C, H, W, O, K, Ho, Wo, pad_h, pad_w, stride_h, stride_w
        );
    } else if (K == 3) {
        bika_conv2d_forward_kernel<3><<<grid, block, shared_bytes>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            scale_ptr, shift_ptr, pweight_ptr, neg_bias_half_ptr, do_relu,
            B, C, H, W, O, K, Ho, Wo, pad_h, pad_w, stride_h, stride_w
        );
    } else {
        bika_conv2d_forward_kernel<0><<<grid, block, shared_bytes>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            scale_ptr, shift_ptr, pweight_ptr, neg_bias_half_ptr, do_relu,
            B, C, H, W, O, K, Ho, Wo, pad_h, pad_w, stride_h, stride_w
        );
    }
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

    auto grad_input  = torch::empty_like(input);
    auto grad_weight = torch::empty_like(weight);
    auto grad_bias   = torch::empty_like(bias);

    const int dev = input.get_device();
    TORCH_CHECK(dev >= 0, "Input must be a CUDA tensor");
    cudaSetDevice(dev);

    dim3 gridWB(O, C);
    int threadsWB = 256;
    size_t sharedWB = 2 * (K * K) * threadsWB * sizeof(float);

    bika_conv2d_backward_wb_kernel<<<gridWB, threadsWB, sharedWB>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        B, C, H, W, O, K, Ho, Wo,
        pad_h, pad_w,
        stride_h, stride_w
    );

    TORCH_CHECK(
        cudaGetLastError() == cudaSuccess,
        "CUDA launch failed (bika_conv2d_backward_wb)."
    );

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
