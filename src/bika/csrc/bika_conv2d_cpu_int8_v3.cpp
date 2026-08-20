// bika_conv2d_cpu_int8_v3.cpp
// ==========================================================================
// Ultra-Fast Int8 AVX2 SIMD Kernel (V3):
// 1. 4-Pixel x 8-Channel 2D Spatial-Channel Register Tiling (TILE_W=4, TILE_O=8)
// 2. Pre-Quantized Int8 Bias (Zero runtime bias overhead)
// 3. Fast Vectorized Float-to-Int8 SIMD conversion
// 4. Zero dynamic memory allocation on the critical path
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
static inline void quantize_f32_to_i8_avx2_fast(
    const float* __restrict__ src,
    int8_t* __restrict__ dst,
    int64_t n,
    float scale_val
) {
    __m256 vscale = _mm256_set1_ps(scale_val);
    int64_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256 f0 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 0), vscale);
        __m256 f1 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 8), vscale);
        __m256 f2 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 16), vscale);
        __m256 f3 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 24), vscale);

        __m256i i0 = _mm256_cvtps_epi32(f0);
        __m256i i1 = _mm256_cvtps_epi32(f1);
        __m256i i2 = _mm256_cvtps_epi32(f2);
        __m256i i3 = _mm256_cvtps_epi32(f3);

        __m256i p01_16 = _mm256_packs_epi32(i0, i1);
        __m256i p23_16 = _mm256_packs_epi32(i2, i3);
        __m256i p0123_8 = _mm256_packs_epi16(p01_16, p23_16);
        __m256i perm = _mm256_permutevar8x32_epi32(p0123_8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        _mm256_storeu_si256((__m256i*)(dst + i), perm);
    }

    for (; i < n; ++i) {
        float v = src[i] * scale_val;
        dst[i] = (int8_t)std::max(-128, std::min(127, (int)std::round(v)));
    }
}

// 4-Pixel x 8-Channel Microkernel (32 output values computed simultaneously)
static inline void bika_kernel_4p8ch_int8_avx2(
    const int8_t* __restrict__ in_p0,
    const int8_t* __restrict__ in_p1,
    const int8_t* __restrict__ in_p2,
    const int8_t* __restrict__ in_p3,
    const int8_t* __restrict__ const* __restrict__ nb_ptrs,
    const unsigned int* __restrict__ const* __restrict__ w_ptrs,
    int ckk,
    int* __restrict__ a0,
    int* __restrict__ a1,
    int* __restrict__ a2,
    int* __restrict__ a3
) {
    #pragma unroll(8)
    for (int ch = 0; ch < 8; ++ch) {
        a0[ch] = 0; a1[ch] = 0; a2[ch] = 0; a3[ch] = 0;
    }
    int i = 0;
    int word = 0;

    for (; i + 32 <= ckk; i += 32, word++) {
        __m256i x0 = _mm256_loadu_si256((const __m256i*)(in_p0 + i));
        __m256i x1 = _mm256_loadu_si256((const __m256i*)(in_p1 + i));
        __m256i x2 = _mm256_loadu_si256((const __m256i*)(in_p2 + i));
        __m256i x3 = _mm256_loadu_si256((const __m256i*)(in_p3 + i));

        #pragma unroll(8)
        for (int ch = 0; ch < 8; ++ch) {
            __m256i nb = _mm256_loadu_si256((const __m256i*)(nb_ptrs[ch] + i));
            unsigned int w_val = w_ptrs[ch][word];

            unsigned int m0 = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nb, x0));
            unsigned int m1 = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nb, x1));
            unsigned int m2 = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nb, x2));
            unsigned int m3 = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nb, x3));

            a0[ch] += 2 * POPCOUNT32(m0 ^ w_val) - 32;
            a1[ch] += 2 * POPCOUNT32(m1 ^ w_val) - 32;
            a2[ch] += 2 * POPCOUNT32(m2 ^ w_val) - 32;
            a3[ch] += 2 * POPCOUNT32(m3 ^ w_val) - 32;
        }
    }

    if (i < ckk) {
        int rem_bits = ckk - i;
        unsigned int bm0[8] = {0}, bm1[8] = {0}, bm2[8] = {0}, bm3[8] = {0};

        for (int k = 0; k < rem_bits; ++k) {
            int8_t xv0 = in_p0[i + k], xv1 = in_p1[i + k], xv2 = in_p2[i + k], xv3 = in_p3[i + k];
            #pragma unroll(8)
            for (int ch = 0; ch < 8; ++ch) {
                int8_t nb_val = nb_ptrs[ch][i + k];
                if (nb_val > xv0) bm0[ch] |= (1u << k);
                if (nb_val > xv1) bm1[ch] |= (1u << k);
                if (nb_val > xv2) bm2[ch] |= (1u << k);
                if (nb_val > xv3) bm3[ch] |= (1u << k);
            }
        }

        unsigned int bit_mask = (rem_bits == 32) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1);
        #pragma unroll(8)
        for (int ch = 0; ch < 8; ++ch) {
            a0[ch] += 2 * POPCOUNT32((bm0[ch] ^ w_ptrs[ch][word]) & bit_mask) - rem_bits;
            a1[ch] += 2 * POPCOUNT32((bm1[ch] ^ w_ptrs[ch][word]) & bit_mask) - rem_bits;
            a2[ch] += 2 * POPCOUNT32((bm2[ch] ^ w_ptrs[ch][word]) & bit_mask) - rem_bits;
            a3[ch] += 2 * POPCOUNT32((bm3[ch] ^ w_ptrs[ch][word]) & bit_mask) - rem_bits;
        }
    }
}
#endif

// ============================================================
// Main Int8 V3 Entrypoint (TILE_W=4, TILE_O=8)
// ============================================================
torch::Tensor bika_conv2d_forward_cpu_int8_v3(
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

    // Pre-Quantized Int8 Bias
    std::vector<int8_t> neg_bias_local;
    const int8_t* nb_i8_ptr;
    if (packed_bias_i8.numel() > 0 && packed_bias_i8.dtype() == torch::kInt8) {
        nb_i8_ptr = packed_bias_i8.contiguous().data_ptr<int8_t>();
    } else {
        const float* b_ptr = bias.contiguous().data_ptr<float>();
        neg_bias_local.resize(O * ckk);
        for (int i = 0; i < O * ckk; ++i) {
            float val = -b_ptr[i] * BIKA_INT8_SCALE;
            neg_bias_local[i] = (int8_t)std::max(-128, std::min(127, (int)std::round(val)));
        }
        nb_i8_ptr = neg_bias_local.data();
    }

    // Fast Vectorized Input Quantization (AVX2 SIMD)
    const int64_t total_in = (int64_t)B * C * H * W;
    std::vector<int8_t> input_i8(total_in);
    #ifdef BIKA_X86
    quantize_f32_to_i8_avx2_fast(in_ptr, input_i8.data(), total_in, BIKA_INT8_SCALE);
    #else
    for (int64_t i = 0; i < total_in; ++i) {
        float val = in_ptr[i] * BIKA_INT8_SCALE;
        input_i8[i] = (int8_t)std::max(-128, std::min(127, (int)std::round(val)));
    }
    #endif
    const int8_t* in_i8_ptr = input_i8.data();

    const int ho_start_safe = (pad_h > 0) ? (pad_h + stride_h - 1) / stride_h : 0;
    const int ho_end_safe = std::min(Ho, (H + pad_h - K) / stride_h + 1);
    const int wo_start_safe = (pad_w > 0) ? (pad_w + stride_w - 1) / stride_w : 0;
    const int wo_end_safe = std::min(Wo, (W + pad_w - K) / stride_w + 1);

    #pragma omp parallel
    {
        alignas(64) int8_t gbuf0[ckk > 0 ? ckk : 1];
        alignas(64) int8_t gbuf1[ckk > 0 ? ckk : 1];
        alignas(64) int8_t gbuf2[ckk > 0 ? ckk : 1];
        alignas(64) int8_t gbuf3[ckk > 0 ? ckk : 1];

        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int b = 0; b < B; ++b) {
            for (int ho = 0; ho < Ho; ++ho) {
                const bool ho_safe = (ho >= ho_start_safe && ho < ho_end_safe);

                int wo = 0;
                // 4 pixels at a time
                for (; wo + 3 < Wo; wo += 4) {
                    const bool all_safe = ho_safe && (wo >= wo_start_safe && (wo + 3) < wo_end_safe);

                    if (all_safe) {
                        int elem = 0;
                        for (int c = 0; c < C; ++c) {
                            const int8_t* in_c = in_i8_ptr + ((int64_t)b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                const int8_t* in_row = in_c + hi * W;
                                const int wi_base = wo * stride_w - pad_w;
                                if (__builtin_expect(K == 3, 1)) {
                                    gbuf0[elem]   = in_row[wi_base];
                                    gbuf0[elem+1] = in_row[wi_base+1];
                                    gbuf0[elem+2] = in_row[wi_base+2];

                                    gbuf1[elem]   = in_row[wi_base+1];
                                    gbuf1[elem+1] = in_row[wi_base+2];
                                    gbuf1[elem+2] = in_row[wi_base+3];

                                    gbuf2[elem]   = in_row[wi_base+2];
                                    gbuf2[elem+1] = in_row[wi_base+3];
                                    gbuf2[elem+2] = in_row[wi_base+4];

                                    gbuf3[elem]   = in_row[wi_base+3];
                                    gbuf3[elem+1] = in_row[wi_base+4];
                                    gbuf3[elem+2] = in_row[wi_base+5];
                                    elem += 3;
                                } else {
                                    for (int kw = 0; kw < K; ++kw) {
                                        gbuf0[elem] = in_row[wi_base + kw];
                                        gbuf1[elem] = in_row[wi_base + 1 + kw];
                                        gbuf2[elem] = in_row[wi_base + 2 + kw];
                                        gbuf3[elem] = in_row[wi_base + 3 + kw];
                                        elem++;
                                    }
                                }
                            }
                        }
                    } else {
                        // Fallback gathering
                        for (int p = 0; p < 4; ++p) {
                            int8_t* gbuf = (p == 0) ? gbuf0 : ((p == 1) ? gbuf1 : ((p == 2) ? gbuf2 : gbuf3));
                            int curr_wo = wo + p;
                            int elem = 0;
                            for (int c = 0; c < C; ++c) {
                                const int8_t* in_c = in_i8_ptr + ((int64_t)b * C + c) * H * W;
                                for (int kh = 0; kh < K; ++kh) {
                                    const int hi = ho * stride_h - pad_h + kh;
                                    if (hi >= 0 && hi < H) {
                                        const int8_t* in_row = in_c + hi * W;
                                        for (int kw = 0; kw < K; ++kw) {
                                            const int wi = curr_wo * stride_w - pad_w + kw;
                                            gbuf[elem++] = (wi >= 0 && wi < W) ? in_row[wi] : (int8_t)0;
                                        }
                                    } else {
                                        for (int kw = 0; kw < K; ++kw) gbuf[elem++] = (int8_t)0;
                                    }
                                }
                            }
                        }
                    }

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
                            w_packed_ptr + (o + 0) * num_words, w_packed_ptr + (o + 1) * num_words,
                            w_packed_ptr + (o + 2) * num_words, w_packed_ptr + (o + 3) * num_words,
                            w_packed_ptr + (o + 4) * num_words, w_packed_ptr + (o + 5) * num_words,
                            w_packed_ptr + (o + 6) * num_words, w_packed_ptr + (o + 7) * num_words
                        };

                        int a0[8], a1[8], a2[8], a3[8];
                        bika_kernel_4p8ch_int8_avx2(gbuf0, gbuf1, gbuf2, gbuf3, nb_ptrs, w_ptrs, ckk, a0, a1, a2, a3);

                        #pragma unroll(8)
                        for (int t = 0; t < 8; ++t) {
                            float r0 = (float)a0[t], r1 = (float)a1[t], r2 = (float)a2[t], r3 = (float)a3[t];
                            if (scale_ptr) {
                                r0 = r0 * scale_ptr[o + t] + shift_ptr[o + t];
                                r1 = r1 * scale_ptr[o + t] + shift_ptr[o + t];
                                r2 = r2 * scale_ptr[o + t] + shift_ptr[o + t];
                                r3 = r3 * scale_ptr[o + t] + shift_ptr[o + t];
                            }
                            if (do_relu) {
                                if (r0 < 0.0f) r0 = 0.0f;
                                if (r1 < 0.0f) r1 = 0.0f;
                                if (r2 < 0.0f) r2 = 0.0f;
                                if (r3 < 0.0f) r3 = 0.0f;
                            }
                            int64_t base_idx = ((int64_t)(b * O + o + t) * Ho + ho) * Wo + wo;
                            out_ptr[base_idx + 0] = r0;
                            out_ptr[base_idx + 1] = r1;
                            out_ptr[base_idx + 2] = r2;
                            out_ptr[base_idx + 3] = r3;
                        }
                    }

                    for (; o < O; ++o) {
                        for (int p = 0; p < 4; ++p) {
                            const int8_t* gbuf = (p == 0) ? gbuf0 : ((p == 1) ? gbuf1 : ((p == 2) ? gbuf2 : gbuf3));
                            int acc = 0, i = 0, word = 0;
                            for (; i + 32 <= ckk; i += 32, word++) {
                                __m256i xv = _mm256_loadu_si256((const __m256i*)(gbuf + i));
                                __m256i nbv = _mm256_loadu_si256((const __m256i*)(nb_i8_ptr + o * ckk + i));
                                unsigned int m = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nbv, xv));
                                acc += 2 * POPCOUNT32(m ^ w_packed_ptr[o * num_words + word]) - 32;
                            }
                            if (i < ckk) {
                                int rem = ckk - i;
                                unsigned int bm = 0;
                                for (int k = 0; k < rem; ++k) {
                                    if (nb_i8_ptr[o * ckk + i + k] > gbuf[i + k]) bm |= (1u << k);
                                }
                                unsigned int bit_mask = (rem == 32) ? 0xFFFFFFFFu : ((1u << rem) - 1);
                                acc += 2 * POPCOUNT32((bm ^ w_packed_ptr[o * num_words + word]) & bit_mask) - rem;
                            }
                            float r = (float)acc;
                            if (scale_ptr) r = r * scale_ptr[o] + shift_ptr[o];
                            if (do_relu && r < 0.0f) r = 0.0f;
                            out_ptr[((int64_t)(b * O + o) * Ho + ho) * Wo + wo + p] = r;
                        }
                    }
                    #endif
                }

                // Remainder 1-3 pixels
                for (; wo < Wo; ++wo) {
                    const bool safe = ho_safe && (wo >= wo_start_safe && wo < wo_end_safe);
                    int elem = 0;
                    if (safe) {
                        for (int c = 0; c < C; ++c) {
                            const int8_t* in_c = in_i8_ptr + ((int64_t)b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                const int8_t* in_row = in_c + hi * W;
                                const int wi_base = wo * stride_w - pad_w;
                                if (K == 3) {
                                    gbuf0[elem]   = in_row[wi_base];
                                    gbuf0[elem+1] = in_row[wi_base+1];
                                    gbuf0[elem+2] = in_row[wi_base+2];
                                    elem += 3;
                                } else {
                                    for (int kw = 0; kw < K; ++kw) gbuf0[elem++] = in_row[wi_base + kw];
                                }
                            }
                        }
                    } else {
                        for (int c = 0; c < C; ++c) {
                            const int8_t* in_c = in_i8_ptr + ((int64_t)b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                if (hi >= 0 && hi < H) {
                                    const int8_t* in_row = in_c + hi * W;
                                    for (int kw = 0; kw < K; ++kw) {
                                        const int wi = wo * stride_w - pad_w + kw;
                                        gbuf0[elem++] = (wi >= 0 && wi < W) ? in_row[wi] : (int8_t)0;
                                    }
                                } else {
                                    for (int kw = 0; kw < K; ++kw) gbuf0[elem++] = (int8_t)0;
                                }
                            }
                        }
                    }

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
                            w_packed_ptr + (o + 0) * num_words, w_packed_ptr + (o + 1) * num_words,
                            w_packed_ptr + (o + 2) * num_words, w_packed_ptr + (o + 3) * num_words,
                            w_packed_ptr + (o + 4) * num_words, w_packed_ptr + (o + 5) * num_words,
                            w_packed_ptr + (o + 6) * num_words, w_packed_ptr + (o + 7) * num_words
                        };
                        int a0[8];
                        int a1_d[8], a2_d[8], a3_d[8];
                        bika_kernel_4p8ch_int8_avx2(gbuf0, gbuf0, gbuf0, gbuf0, nb_ptrs, w_ptrs, ckk, a0, a1_d, a2_d, a3_d);
                        #pragma unroll(8)
                        for (int t = 0; t < 8; ++t) {
                            float r = (float)a0[t];
                            if (scale_ptr) r = r * scale_ptr[o + t] + shift_ptr[o + t];
                            if (do_relu && r < 0.0f) r = 0.0f;
                            out_ptr[((int64_t)(b * O + o + t) * Ho + ho) * Wo + wo] = r;
                        }
                    }
                    for (; o < O; ++o) {
                        int acc = 0, i = 0, word = 0;
                        for (; i + 32 <= ckk; i += 32, word++) {
                            __m256i xv = _mm256_loadu_si256((const __m256i*)(gbuf0 + i));
                            __m256i nbv = _mm256_loadu_si256((const __m256i*)(nb_i8_ptr + o * ckk + i));
                            unsigned int m = (unsigned int)_mm256_movemask_epi8(_mm256_cmpgt_epi8(nbv, xv));
                            acc += 2 * POPCOUNT32(m ^ w_packed_ptr[o * num_words + word]) - 32;
                        }
                        if (i < ckk) {
                            int rem = ckk - i;
                            unsigned int bm = 0;
                            for (int k = 0; k < rem; ++k) {
                                if (nb_i8_ptr[o * ckk + i + k] > gbuf0[i + k]) bm |= (1u << k);
                            }
                            unsigned int bit_mask = (rem == 32) ? 0xFFFFFFFFu : ((1u << rem) - 1);
                            acc += 2 * POPCOUNT32((bm ^ w_packed_ptr[o * num_words + word]) & bit_mask) - rem;
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
