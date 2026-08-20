// bika_conv2d_cpu_nhwc.cpp
// ==========================================================================
// Strategy 4 (Advanced): Channels-Last (NHWC) Contiguous Streaming
// Int8 AVX2 SIMD Microkernel for BiKA BNN Binary Convolutions
//
// Key Innovation:
// - Transforms NCHW input to NHWC (Channels-Last) contiguous format.
// - In NHWC, 3x3 patch loading becomes 3 sequential 128/256-bit vector loads
//   instead of C*3 scattered scalar pointer lookups (24x fewer load instructions).
// - Combines with 8-channel AVX2 microkernel and pre-quantized int8 bias.
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

#ifdef BIKA_X86
// 8-Channel AVX2 Int8 Microkernel
static inline void bika_nhwc_8ch_kernel(
    const int8_t* __restrict__ in_patch,
    const int8_t* __restrict__ const* __restrict__ nb_ptrs,
    const unsigned int* __restrict__ const* __restrict__ w_ptrs,
    int ckk,
    int* __restrict__ acc_out
) {
    int a[8] = {0};
    int i = 0;
    int word = 0;

    for (; i + 32 <= ckk; i += 32, word++) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(in_patch + i));
        #pragma unroll(8)
        for (int ch = 0; ch < 8; ++ch) {
            __m256i nb = _mm256_loadu_si256((const __m256i*)(nb_ptrs[ch] + i));
            unsigned int m = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nb, x));
            a[ch] += 2 * POPCOUNT32(m ^ w_ptrs[ch][word]) - 32;
        }
    }

    if (i < ckk) {
        int rem_bits = ckk - i;
        unsigned int bm[8] = {0};
        for (int k = 0; k < rem_bits; ++k) {
            int8_t xv = in_patch[i + k];
            #pragma unroll(8)
            for (int ch = 0; ch < 8; ++ch) {
                if (nb_ptrs[ch][i + k] > xv) bm[ch] |= (1u << k);
            }
        }
        unsigned int bit_mask = (rem_bits == 32) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1);
        #pragma unroll(8)
        for (int ch = 0; ch < 8; ++ch) {
            a[ch] += 2 * POPCOUNT32((bm[ch] ^ w_ptrs[ch][word]) & bit_mask) - rem_bits;
        }
    }

    #pragma unroll(8)
    for (int ch = 0; ch < 8; ++ch) {
        acc_out[ch] = a[ch];
    }
}
#endif

// ============================================================
// Main NHWC Forward Function
// ============================================================
torch::Tensor bika_conv2d_forward_cpu_nhwc(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out_scale,
    torch::Tensor out_shift,
    torch::Tensor packed_weight,
    torch::Tensor packed_bias_i8,
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
    float* out_ptr = output.data_ptr<float>();

    const float* scale_ptr = (out_scale.numel() > 0) ? out_scale.contiguous().data_ptr<float>() : nullptr;
    const float* shift_ptr = (out_shift.numel() > 0) ? out_shift.contiguous().data_ptr<float>() : nullptr;

    const int ckk = C * K * K;
    const int num_words = (ckk + 31) / 32;

    // Packed Weights
    const int* pw_ptr = (packed_weight.numel() > 0) ? packed_weight.contiguous().data_ptr<int>() : nullptr;
    std::vector<unsigned int> w_packed_local;
    const unsigned int* w_packed_ptr;
    if (pw_ptr) {
        w_packed_ptr = reinterpret_cast<const unsigned int*>(pw_ptr);
    } else {
        const float* w_ptr = weight.contiguous().data_ptr<float>();
        w_packed_local.resize(O * num_words, 0);
        for (int o = 0; o < O; ++o) {
            for (int i = 0; i < ckk; ++i) {
                int word = i / 32, bit = i % 32;
                if (w_ptr[o * ckk + i] >= 0.0f) w_packed_local[o * num_words + word] |= (1u << bit);
            }
        }
        w_packed_ptr = w_packed_local.data();
    }

    // Pre-Quantized Int8 Bias (Reordered for NHWC contiguous matching)
    // In standard NCHW, bias layout is [O, C, Kh, Kw].
    // For NHWC patch streaming, patch is ordered as [Kh, Kw, C].
    // We reorder bias once to [O, Kh, Kw, C] so vector loads match 100%!
    std::vector<int8_t> neg_bias_nhwc(O * ckk);
    const float* b_ptr = bias.contiguous().data_ptr<float>();
    for (int o = 0; o < O; ++o) {
        int dst_idx = 0;
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                for (int c = 0; c < C; ++c) {
                    int src_idx = o * ckk + c * (K * K) + kh * K + kw;
                    float val = -b_ptr[src_idx] * BIKA_INT8_SCALE;
                    neg_bias_nhwc[o * ckk + dst_idx] = (int8_t)std::max(-128, std::min(127, (int)std::round(val)));
                    dst_idx++;
                }
            }
        }
    }
    const int8_t* nb_i8_ptr = neg_bias_nhwc.data();

    // Reorder packed weights to match NHWC patch layout [Kh, Kw, C]
    std::vector<unsigned int> w_packed_nhwc(O * num_words, 0);
    const float* w_raw_ptr = weight.contiguous().data_ptr<float>();
    for (int o = 0; o < O; ++o) {
        int dst_idx = 0;
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                for (int c = 0; c < C; ++c) {
                    int src_idx = o * ckk + c * (K * K) + kh * K + kw;
                    if (w_raw_ptr[src_idx] >= 0.0f) {
                        int word = dst_idx / 32, bit = dst_idx % 32;
                        w_packed_nhwc[o * num_words + word] |= (1u << bit);
                    }
                    dst_idx++;
                }
            }
        }
    }
    const unsigned int* w_nhwc_ptr = w_packed_nhwc.data();

    // 1. Fast NCHW -> NHWC Int8 Quantized Layout Transformation
    // Transposes [B, C, H, W] Float32 directly into [B, H, W, C] Int8!
    const int64_t total_in = (int64_t)B * H * W * C;
    std::vector<int8_t> input_nhwc(total_in);
    int8_t* in_nhwc_ptr = input_nhwc.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int8_t* dst_pixel = in_nhwc_ptr + (((int64_t)b * H + h) * W + w) * C;
                for (int c = 0; c < C; ++c) {
                    float val = in_ptr[(((int64_t)b * C + c) * H + h) * W + w] * BIKA_INT8_SCALE;
                    dst_pixel[c] = (int8_t)std::max(-128, std::min(127, (int)std::round(val)));
                }
            }
        }
    }

    const int ho_start_safe = (pad_h > 0) ? (pad_h + stride_h - 1) / stride_h : 0;
    const int ho_end_safe = std::min(Ho, (H + pad_h - K) / stride_h + 1);
    const int wo_start_safe = (pad_w > 0) ? (pad_w + stride_w - 1) / stride_w : 0;
    const int wo_end_safe = std::min(Wo, (W + pad_w - K) / stride_w + 1);

    // 2. NHWC Parallel Convolution with Fast Contiguous Row Streaming
    #pragma omp parallel
    {
        alignas(64) int8_t patch_buf[ckk > 0 ? ckk : 1];

        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int b = 0; b < B; ++b) {
            for (int ho = 0; ho < Ho; ++ho) {
                const bool ho_safe = (ho >= ho_start_safe && ho < ho_end_safe);
                const int8_t* in_b = in_nhwc_ptr + (int64_t)b * H * W * C;

                for (int wo = 0; wo < Wo; ++wo) {
                    const bool safe = ho_safe && (wo >= wo_start_safe && wo < wo_end_safe);

                    if (__builtin_expect(safe, 1)) {
                        // FAST NHWC STREAMING: Contiguous row copies!
                        int elem = 0;
                        for (int kh = 0; kh < K; ++kh) {
                            const int hi = ho * stride_h - pad_h + kh;
                            const int wi = wo * stride_w - pad_w;
                            const int8_t* src_row = in_b + ((int64_t)hi * W + wi) * C;
                            std::memcpy(patch_buf + elem, src_row, K * C);
                            elem += K * C;
                        }
                    } else {
                        int elem = 0;
                        for (int kh = 0; kh < K; ++kh) {
                            const int hi = ho * stride_h - pad_h + kh;
                            for (int kw = 0; kw < K; ++kw) {
                                const int wi = wo * stride_w - pad_w + kw;
                                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                    const int8_t* src_p = in_b + ((int64_t)hi * W + wi) * C;
                                    std::memcpy(patch_buf + elem, src_p, C);
                                } else {
                                    std::memset(patch_buf + elem, 0, C);
                                }
                                elem += C;
                            }
                        }
                    }

                    // 8-Channel Vectorized Microkernel
                    int o = 0;
                    #ifdef BIKA_X86
                    for (; o + 8 <= O; o += 8) {
                        const int8_t* nb_ptrs[8] = {
                            nb_i8_ptr + (o + 0) * ckk, nb_i8_ptr + (o + 1) * ckk,
                            nb_i8_ptr + (o + 2) * ckk, nb_i8_ptr + (o + 3) * ckk,
                            nb_i8_ptr + (o + 4) * ckk, nb_i8_ptr + (o + 5) * ckk,
                            nb_i8_ptr + (o + 6) * ckk, nb_i8_ptr + (o + 7) * ckk
                        };
                        const unsigned int* w_ptrs[8] = {
                            w_nhwc_ptr + (o + 0) * num_words, w_nhwc_ptr + (o + 1) * num_words,
                            w_nhwc_ptr + (o + 2) * num_words, w_nhwc_ptr + (o + 3) * num_words,
                            w_nhwc_ptr + (o + 4) * num_words, w_nhwc_ptr + (o + 5) * num_words,
                            w_nhwc_ptr + (o + 6) * num_words, w_nhwc_ptr + (o + 7) * num_words
                        };

                        int acc[8];
                        bika_nhwc_8ch_kernel(patch_buf, nb_ptrs, w_ptrs, ckk, acc);

                        #pragma unroll(8)
                        for (int t = 0; t < 8; ++t) {
                            float r = (float)acc[t];
                            if (scale_ptr) r = r * scale_ptr[o + t] + shift_ptr[o + t];
                            if (do_relu && r < 0.0f) r = 0.0f;
                            out_ptr[((int64_t)(b * O + o + t) * Ho + ho) * Wo + wo] = r;
                        }
                    }

                    for (; o < O; ++o) {
                        int acc = 0, i = 0, word = 0;
                        for (; i + 32 <= ckk; i += 32, word++) {
                            __m256i xv = _mm256_loadu_si256((const __m256i*)(patch_buf + i));
                            __m256i nbv = _mm256_loadu_si256((const __m256i*)(nb_i8_ptr + o * ckk + i));
                            unsigned int m = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nbv, xv));
                            acc += 2 * POPCOUNT32(m ^ w_nhwc_ptr[o * num_words + word]) - 32;
                        }
                        if (i < ckk) {
                            int rem = ckk - i;
                            unsigned int bm = 0;
                            for (int k = 0; k < rem; ++k) {
                                if (nb_i8_ptr[o * ckk + i + k] > patch_buf[i + k]) bm |= (1u << k);
                            }
                            unsigned int bit_mask = (rem == 32) ? 0xFFFFFFFFu : ((1u << rem) - 1);
                            acc += 2 * POPCOUNT32((bm ^ w_nhwc_ptr[o * num_words + word]) & bit_mask) - rem;
                        }
                        float r = (float)acc;
                        if (scale_ptr) r = r * scale_ptr[o] + shift_ptr[o];
                        if (do_relu && r < 0.0f) r = 0.0f;
                        out_ptr[((int64_t)(b * O + o) * Ho + ho) * Wo + wo] = r;
                    }
                    #endif
                }
            }
        }
    }

    return output;
}
