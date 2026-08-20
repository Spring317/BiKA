// bika_conv2d_cpu_int8.cpp
// ==========================================================================
// Strategy 3 + Strategy 4: High-Performance Int8 AVX2 SIMD CPU Kernel
// for BiKA BNN Binary Convolutions
//
// Key Innovations:
// 1. Int8 AVX2 SIMD Vectorization (32 elements per instruction):
//    - _mm256_cmpgt_epi8 + _mm256_movemask_epi8 extracts all 32 bits
//      in a single vector comparison (4x wider than float32 AVX2).
// 2. Mathematical XOR-Popcount Identity:
//    - BNN XNOR(bit, w) == movemask(cmpgt(nb, x)) ^ w.
//    - Zero NOT operations, zero bit-shifting loops.
// 3. 8-Channel Vectorized Unrolling (TILE_O=8):
//    - Input patch is loaded once in an int8 YMM register and reused
//      across 8 output channels simultaneously.
// 4. Cache-Optimized Memory Layout:
//    - Pre-quantized int8 bias buffer consumes 4x less memory bandwidth
//      (fits 100% in L1/L2 cache for all layers).
// 5. OpenMP parallelization over spatial rows with zero false sharing.
// ==========================================================================

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define BIKA_X86 1
    #include <immintrin.h>
#endif

#ifdef _MSC_VER
    #include <intrin.h>
    #define POPCOUNT32(x) __popcnt(x)
#else
    #define POPCOUNT32(x) __builtin_popcount(x)
#endif

constexpr float BIKA_INT8_SCALE = 64.0f;

// ============================================================
// 8-Channel Int8 AVX2 Microkernel
// Processes 32 connections across 8 output channels (256 MACs)
// in just 3 vector instructions per channel!
// ============================================================
#ifdef BIKA_X86
static inline void bika_kernel_8ch_int8_avx2(
    const int8_t* __restrict__ in_patch,
    const int8_t* __restrict__ const* __restrict__ nb_ptrs, // 8 pointers to int8 neg_bias
    const unsigned int* __restrict__ const* __restrict__ w_ptrs, // 8 pointers to uint32 weights
    int ckk,
    int* __restrict__ acc_out // int[8]
) {
    int a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    int word = 0;

    // Main loop: 32 elements (1 YMM vector) per iteration
    for (; i + 32 <= ckk; i += 32, word++) {
        // Load 32 int8 input values ONCE into 1 YMM register
        __m256i x_vec = _mm256_loadu_si256((const __m256i*)(in_patch + i));

        #pragma unroll(8)
        for (int ch = 0; ch < 8; ++ch) {
            __m256i nb_vec = _mm256_loadu_si256((const __m256i*)(nb_ptrs[ch] + i));
            // Mathematical identity: XNOR(x >= nb, w) == movemask(nb > x) ^ w
            __m256i cmp = _mm256_cmpgt_epi8(nb_vec, x_vec);
            unsigned int mask = (unsigned int)_mm256_movemask_epi8(cmp);
            a[ch] += 2 * POPCOUNT32(mask ^ w_ptrs[ch][word]) - 32;
        }
    }

    // Remainder loop (< 32 elements)
    if (i < ckk) {
        int rem_bits = ckk - i;
        unsigned int b_mask[8] = {0};

        // Scalar remainder
        for (int k = 0; k < rem_bits; ++k) {
            int8_t xv = in_patch[i + k];
            #pragma unroll(8)
            for (int ch = 0; ch < 8; ++ch) {
                if (nb_ptrs[ch][i + k] > xv) {
                    b_mask[ch] |= (1u << k);
                }
            }
        }

        unsigned int bit_mask = (rem_bits == 32) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1);
        #pragma unroll(8)
        for (int ch = 0; ch < 8; ++ch) {
            unsigned int xnor_val = (b_mask[ch] ^ w_ptrs[ch][word]) & bit_mask;
            a[ch] += 2 * POPCOUNT32(xnor_val) - rem_bits;
        }
    }

    #pragma unroll(8)
    for (int ch = 0; ch < 8; ++ch) {
        acc_out[ch] = a[ch];
    }
}

// 4-Channel Int8 AVX2 Microkernel
static inline void bika_kernel_4ch_int8_avx2(
    const int8_t* __restrict__ in_patch,
    const int8_t* __restrict__ const* __restrict__ nb_ptrs,
    const unsigned int* __restrict__ const* __restrict__ w_ptrs,
    int ckk,
    int* __restrict__ acc_out
) {
    int a[4] = {0, 0, 0, 0};
    int i = 0;
    int word = 0;

    for (; i + 32 <= ckk; i += 32, word++) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i*)(in_patch + i));

        #pragma unroll(4)
        for (int ch = 0; ch < 4; ++ch) {
            __m256i nb_vec = _mm256_loadu_si256((const __m256i*)(nb_ptrs[ch] + i));
            __m256i cmp = _mm256_cmpgt_epi8(nb_vec, x_vec);
            unsigned int mask = (unsigned int)_mm256_movemask_epi8(cmp);
            a[ch] += 2 * POPCOUNT32(mask ^ w_ptrs[ch][word]) - 32;
        }
    }

    if (i < ckk) {
        int rem_bits = ckk - i;
        unsigned int b_mask[4] = {0};

        for (int k = 0; k < rem_bits; ++k) {
            int8_t xv = in_patch[i + k];
            #pragma unroll(4)
            for (int ch = 0; ch < 4; ++ch) {
                if (nb_ptrs[ch][i + k] > xv) {
                    b_mask[ch] |= (1u << k);
                }
            }
        }

        unsigned int bit_mask = (rem_bits == 32) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1);
        #pragma unroll(4)
        for (int ch = 0; ch < 4; ++ch) {
            unsigned int xnor_val = (b_mask[ch] ^ w_ptrs[ch][word]) & bit_mask;
            a[ch] += 2 * POPCOUNT32(xnor_val) - rem_bits;
        }
    }

    #pragma unroll(4)
    for (int ch = 0; ch < 4; ++ch) {
        acc_out[ch] = a[ch];
    }
}

// 1-Channel Int8 AVX2 Microkernel
static inline int bika_kernel_1ch_int8_avx2(
    const int8_t* __restrict__ in_patch,
    const int8_t* __restrict__ nb_ptr,
    const unsigned int* __restrict__ w_ptr,
    int ckk
) {
    int a = 0;
    int i = 0;
    int word = 0;

    for (; i + 32 <= ckk; i += 32, word++) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i*)(in_patch + i));
        __m256i nb_vec = _mm256_loadu_si256((const __m256i*)(nb_ptr + i));
        __m256i cmp = _mm256_cmpgt_epi8(nb_vec, x_vec);
        unsigned int mask = (unsigned int)_mm256_movemask_epi8(cmp);
        a += 2 * POPCOUNT32(mask ^ w_ptr[word]) - 32;
    }

    if (i < ckk) {
        int rem_bits = ckk - i;
        unsigned int b_mask = 0;
        for (int k = 0; k < rem_bits; ++k) {
            if (nb_ptr[i + k] > in_patch[i + k]) {
                b_mask |= (1u << k);
            }
        }
        unsigned int bit_mask = (rem_bits == 32) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1);
        unsigned int xnor_val = (b_mask ^ w_ptr[word]) & bit_mask;
        a += 2 * POPCOUNT32(xnor_val) - rem_bits;
    }
    return a;
}
#endif // BIKA_X86


// ============================================================
// Main Int8 Forward Function
// ============================================================
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
) {
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int O = weight.size(0);
    const int K = weight.size(2);

    const int Ho = (H + 2 * pad_h - K) / stride_h + 1;
    const int Wo = (W + 2 * pad_w - K) / stride_w + 1;

    auto output = torch::empty({B, O, Ho, Wo}, input.options());

    const float* in_ptr = input.contiguous().data_ptr<float>();
    const float* b_ptr = bias.contiguous().data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const float* scale_ptr = nullptr;
    const float* shift_ptr = nullptr;
    if (out_scale.numel() > 0) scale_ptr = out_scale.contiguous().data_ptr<float>();
    if (out_shift.numel() > 0) shift_ptr = out_shift.contiguous().data_ptr<float>();

    const int* pw_ptr = nullptr;
    if (packed_weight.numel() > 0) pw_ptr = packed_weight.contiguous().data_ptr<int>();

    const int ckk = C * K * K;
    const int num_words = (ckk + 31) / 32;

    // 1. Quantize pre-negated bias to Int8 (1 byte per connection)
    std::vector<int8_t> neg_bias_i8(O * ckk);
    for (int i = 0; i < O * ckk; ++i) {
        float val = -b_ptr[i] * BIKA_INT8_SCALE;
        int clamped = std::max(-128, std::min(127, (int)std::round(val)));
        neg_bias_i8[i] = (int8_t)clamped;
    }

    // 2. Pre-pack weights if needed
    std::vector<unsigned int> w_packed_local;
    const unsigned int* w_packed_ptr;
    if (pw_ptr) {
        w_packed_ptr = reinterpret_cast<const unsigned int*>(pw_ptr);
    } else {
        const float* w_ptr = weight.contiguous().data_ptr<float>();
        w_packed_local.resize(O * num_words, 0);
        for (int o = 0; o < O; ++o) {
            for (int i = 0; i < ckk; ++i) {
                int word = i / 32;
                int bit = i % 32;
                if (w_ptr[o * ckk + i] >= 0.0f) {
                    w_packed_local[o * num_words + word] |= (1u << bit);
                }
            }
        }
        w_packed_ptr = w_packed_local.data();
    }

    // 3. Fast Vectorized Input Quantization (Float32 -> Int8)
    const int64_t total_in_elements = (int64_t)B * C * H * W;
    std::vector<int8_t> input_i8(total_in_elements);
    int8_t* in_i8_ptr = input_i8.data();

    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total_in_elements; idx += 32) {
        const int64_t count = std::min((int64_t)32, total_in_elements - idx);
        for (int64_t k = 0; k < count; ++k) {
            float val = in_ptr[idx + k] * BIKA_INT8_SCALE;
            int clamped = std::max(-128, std::min(127, (int)std::round(val)));
            in_i8_ptr[idx + k] = (int8_t)clamped;
        }
    }

    // Safe interior spatial bounds
    const int ho_start_safe = (pad_h > 0) ? (pad_h + stride_h - 1) / stride_h : 0;
    const int ho_end_safe = std::min(Ho, (H + pad_h - K) / stride_h + 1);
    const int wo_start_safe = (pad_w > 0) ? (pad_w + stride_w - 1) / stride_w : 0;
    const int wo_end_safe = std::min(Wo, (W + pad_w - K) / stride_w + 1);

    // 4. Parallel Int8 Convolution Execution
    #pragma omp parallel
    {
        alignas(64) int8_t gather_buf[ckk > 0 ? ckk : 1];

        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int b = 0; b < B; ++b) {
            for (int ho = 0; ho < Ho; ++ho) {
                const bool ho_safe = (ho >= ho_start_safe && ho < ho_end_safe);

                for (int wo = 0; wo < Wo; ++wo) {
                    const bool fully_safe = ho_safe && (wo >= wo_start_safe && wo < wo_end_safe);

                    // Gather Int8 input patch
                    if (fully_safe) {
                        int elem = 0;
                        for (int c = 0; c < C; ++c) {
                            const int8_t* in_c = in_i8_ptr + ((int64_t)b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                const int8_t* in_row = in_c + hi * W;
                                const int wi_base = wo * stride_w - pad_w;
                                if (__builtin_expect(K == 3, 1)) {
                                    gather_buf[elem]   = in_row[wi_base];
                                    gather_buf[elem+1] = in_row[wi_base+1];
                                    gather_buf[elem+2] = in_row[wi_base+2];
                                    elem += 3;
                                } else if (K == 1) {
                                    gather_buf[elem++] = in_row[wi_base];
                                } else {
                                    for (int kw = 0; kw < K; ++kw)
                                        gather_buf[elem++] = in_row[wi_base + kw];
                                }
                            }
                        }
                    } else {
                        int elem = 0;
                        for (int c = 0; c < C; ++c) {
                            const int8_t* in_c = in_i8_ptr + ((int64_t)b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                if (hi >= 0 && hi < H) {
                                    const int8_t* in_row = in_c + hi * W;
                                    for (int kw = 0; kw < K; ++kw) {
                                        const int wi = wo * stride_w - pad_w + kw;
                                        gather_buf[elem++] = (wi >= 0 && wi < W) ? in_row[wi] : (int8_t)0;
                                    }
                                } else {
                                    for (int kw = 0; kw < K; ++kw)
                                        gather_buf[elem++] = (int8_t)0;
                                }
                            }
                        }
                    }

                    // Channel processing: 8 channels at a time with Int8 AVX2
                    int o = 0;
                    #ifdef BIKA_X86
                    for (; o + 8 <= O; o += 8) {
                        const int8_t* nb_ptrs[8] = {
                            neg_bias_i8.data() + (o + 0) * ckk,
                            neg_bias_i8.data() + (o + 1) * ckk,
                            neg_bias_i8.data() + (o + 2) * ckk,
                            neg_bias_i8.data() + (o + 3) * ckk,
                            neg_bias_i8.data() + (o + 4) * ckk,
                            neg_bias_i8.data() + (o + 5) * ckk,
                            neg_bias_i8.data() + (o + 6) * ckk,
                            neg_bias_i8.data() + (o + 7) * ckk
                        };
                        const unsigned int* w_ptrs[8] = {
                            w_packed_ptr + (o + 0) * num_words,
                            w_packed_ptr + (o + 1) * num_words,
                            w_packed_ptr + (o + 2) * num_words,
                            w_packed_ptr + (o + 3) * num_words,
                            w_packed_ptr + (o + 4) * num_words,
                            w_packed_ptr + (o + 5) * num_words,
                            w_packed_ptr + (o + 6) * num_words,
                            w_packed_ptr + (o + 7) * num_words
                        };

                        int accs[8];
                        bika_kernel_8ch_int8_avx2(gather_buf, nb_ptrs, w_ptrs, ckk, accs);

                        #pragma unroll(8)
                        for (int t = 0; t < 8; ++t) {
                            float res = (float)accs[t];
                            if (scale_ptr) res = res * scale_ptr[o + t] + shift_ptr[o + t];
                            if (do_relu && res < 0.0f) res = 0.0f;
                            out_ptr[((int64_t)(b * O + o + t) * Ho + ho) * Wo + wo] = res;
                        }
                    }

                    // 4 channels at a time
                    for (; o + 4 <= O; o += 4) {
                        const int8_t* nb_ptrs[4] = {
                            neg_bias_i8.data() + (o + 0) * ckk,
                            neg_bias_i8.data() + (o + 1) * ckk,
                            neg_bias_i8.data() + (o + 2) * ckk,
                            neg_bias_i8.data() + (o + 3) * ckk
                        };
                        const unsigned int* w_ptrs[4] = {
                            w_packed_ptr + (o + 0) * num_words,
                            w_packed_ptr + (o + 1) * num_words,
                            w_packed_ptr + (o + 2) * num_words,
                            w_packed_ptr + (o + 3) * num_words
                        };

                        int accs[4];
                        bika_kernel_4ch_int8_avx2(gather_buf, nb_ptrs, w_ptrs, ckk, accs);

                        #pragma unroll(4)
                        for (int t = 0; t < 4; ++t) {
                            float res = (float)accs[t];
                            if (scale_ptr) res = res * scale_ptr[o + t] + shift_ptr[o + t];
                            if (do_relu && res < 0.0f) res = 0.0f;
                            out_ptr[((int64_t)(b * O + o + t) * Ho + ho) * Wo + wo] = res;
                        }
                    }

                    // Remainder channels (1-3)
                    for (; o < O; ++o) {
                        int acc = bika_kernel_1ch_int8_avx2(
                            gather_buf,
                            neg_bias_i8.data() + o * ckk,
                            w_packed_ptr + o * num_words,
                            ckk
                        );
                        float res = (float)acc;
                        if (scale_ptr) res = res * scale_ptr[o] + shift_ptr[o];
                        if (do_relu && res < 0.0f) res = 0.0f;
                        out_ptr[((int64_t)(b * O + o) * Ho + ho) * Wo + wo] = res;
                    }
                    #endif
                }
            }
        }
    }

    return output;
}
