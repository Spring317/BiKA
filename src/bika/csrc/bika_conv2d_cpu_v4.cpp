// bika_conv2d_cpu_v4.cpp
// ==========================================================================
// V4 Ultra-Optimized x86 CPU Kernel for BiKA BNN Binary Convolutions
//
// Key Optimizations:
// 1. Straight-line 32-float AVX2 inner loop (4x load + 4x cmp + 4x movemask)
//    - Zero branch/overflow tracking overhead (exact 32-bit word extraction)
// 2. 8-channel simultaneous processing (TILE_O=8):
//    - Input patch is loaded ONCE and reused across 8 output channels
//    - 8x reduction in input bandwidth
// 3. 4-channel fallback for remaining channels
// 4. L1 Cache-resident bias access:
//    - Contiguous pre-negated bias layout
// 5. OpenMP parallelization across batch and spatial dimensions
// ==========================================================================

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>
#include <algorithm>
#include <cstring>

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

// ============================================================
// 8-channel unrolled AVX2 microkernel
// Reuses 32 input floats across 8 output channels
// ============================================================
#ifdef BIKA_X86
static inline void bika_kernel_8ch_avx2(
    const float* __restrict__ in_patch,
    const float* __restrict__ const* __restrict__ nb, // 8 pointers to neg_bias
    const unsigned int* __restrict__ const* __restrict__ w, // 8 pointers to packed weights
    int ckk,
    int* __restrict__ acc_out // int[8]
) {
    int a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    int word = 0;

    // Main 32-float block loop: processes 32 floats across 8 channels (256 operations per iteration)
    for (; i + 32 <= ckk; i += 32, word++) {
        // Load 32 floats ONCE into 4 YMM registers
        __m256 x0 = _mm256_loadu_ps(in_patch + i + 0);
        __m256 x1 = _mm256_loadu_ps(in_patch + i + 8);
        __m256 x2 = _mm256_loadu_ps(in_patch + i + 16);
        __m256 x3 = _mm256_loadu_ps(in_patch + i + 24);

        #pragma unroll(8)
        for (int ch = 0; ch < 8; ++ch) {
            const float* nb_ch = nb[ch];
            unsigned int m0 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x0, _mm256_loadu_ps(nb_ch + i + 0), _CMP_GE_OQ));
            unsigned int m1 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x1, _mm256_loadu_ps(nb_ch + i + 8), _CMP_GE_OQ));
            unsigned int m2 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x2, _mm256_loadu_ps(nb_ch + i + 16), _CMP_GE_OQ));
            unsigned int m3 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x3, _mm256_loadu_ps(nb_ch + i + 24), _CMP_GE_OQ));

            unsigned int bits32 = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);
            a[ch] += 2 * POPCOUNT32(~(bits32 ^ w[ch][word])) - 32;
        }
    }

    // Tail loop for remaining elements (< 32)
    if (i < ckk) {
        unsigned int b_acc[8] = {0};
        int bit = 0;

        for (; i + 8 <= ckk; i += 8) {
            __m256 x = _mm256_loadu_ps(in_patch + i);
            #pragma unroll(8)
            for (int ch = 0; ch < 8; ++ch) {
                unsigned int m = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x, _mm256_loadu_ps(nb[ch] + i), _CMP_GE_OQ));
                b_acc[ch] |= (m << bit);
            }
            bit += 8;
        }

        for (; i + 4 <= ckk; i += 4) {
            __m128 x = _mm_loadu_ps(in_patch + i);
            #pragma unroll(8)
            for (int ch = 0; ch < 8; ++ch) {
                unsigned int m = (unsigned int)_mm_movemask_ps(_mm_cmpge_ps(x, _mm_loadu_ps(nb[ch] + i)));
                b_acc[ch] |= (m << bit);
            }
            bit += 4;
        }

        for (; i < ckk; ++i) {
            float xv = in_patch[i];
            #pragma unroll(8)
            for (int ch = 0; ch < 8; ++ch) {
                if (xv >= nb[ch][i]) b_acc[ch] |= (1u << bit);
            }
            bit++;
        }

        if (bit > 0) {
            if (bit == 32) {
                #pragma unroll(8)
                for (int ch = 0; ch < 8; ++ch) {
                    a[ch] += 2 * POPCOUNT32(~(b_acc[ch] ^ w[ch][word])) - 32;
                }
            } else {
                unsigned int mask = (1u << bit) - 1;
                #pragma unroll(8)
                for (int ch = 0; ch < 8; ++ch) {
                    a[ch] += 2 * POPCOUNT32((~(b_acc[ch] ^ w[ch][word])) & mask) - bit;
                }
            }
        }
    }

    #pragma unroll(8)
    for (int ch = 0; ch < 8; ++ch) {
        acc_out[ch] = a[ch];
    }
}

// 4-channel unrolled AVX2 microkernel
static inline void bika_kernel_4ch_avx2(
    const float* __restrict__ in_patch,
    const float* __restrict__ const* __restrict__ nb, // 4 pointers
    const unsigned int* __restrict__ const* __restrict__ w, // 4 pointers
    int ckk,
    int* __restrict__ acc_out // int[4]
) {
    int a[4] = {0, 0, 0, 0};
    int i = 0;
    int word = 0;

    for (; i + 32 <= ckk; i += 32, word++) {
        __m256 x0 = _mm256_loadu_ps(in_patch + i + 0);
        __m256 x1 = _mm256_loadu_ps(in_patch + i + 8);
        __m256 x2 = _mm256_loadu_ps(in_patch + i + 16);
        __m256 x3 = _mm256_loadu_ps(in_patch + i + 24);

        #pragma unroll(4)
        for (int ch = 0; ch < 4; ++ch) {
            const float* nb_ch = nb[ch];
            unsigned int m0 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x0, _mm256_loadu_ps(nb_ch + i + 0), _CMP_GE_OQ));
            unsigned int m1 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x1, _mm256_loadu_ps(nb_ch + i + 8), _CMP_GE_OQ));
            unsigned int m2 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x2, _mm256_loadu_ps(nb_ch + i + 16), _CMP_GE_OQ));
            unsigned int m3 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x3, _mm256_loadu_ps(nb_ch + i + 24), _CMP_GE_OQ));

            unsigned int bits32 = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);
            a[ch] += 2 * POPCOUNT32(~(bits32 ^ w[ch][word])) - 32;
        }
    }

    if (i < ckk) {
        unsigned int b_acc[4] = {0};
        int bit = 0;

        for (; i + 8 <= ckk; i += 8) {
            __m256 x = _mm256_loadu_ps(in_patch + i);
            #pragma unroll(4)
            for (int ch = 0; ch < 4; ++ch) {
                unsigned int m = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x, _mm256_loadu_ps(nb[ch] + i), _CMP_GE_OQ));
                b_acc[ch] |= (m << bit);
            }
            bit += 8;
        }

        for (; i + 4 <= ckk; i += 4) {
            __m128 x = _mm_loadu_ps(in_patch + i);
            #pragma unroll(4)
            for (int ch = 0; ch < 4; ++ch) {
                unsigned int m = (unsigned int)_mm_movemask_ps(_mm_cmpge_ps(x, _mm_loadu_ps(nb[ch] + i)));
                b_acc[ch] |= (m << bit);
            }
            bit += 4;
        }

        for (; i < ckk; ++i) {
            float xv = in_patch[i];
            #pragma unroll(4)
            for (int ch = 0; ch < 4; ++ch) {
                if (xv >= nb[ch][i]) b_acc[ch] |= (1u << bit);
            }
            bit++;
        }

        if (bit > 0) {
            if (bit == 32) {
                #pragma unroll(4)
                for (int ch = 0; ch < 4; ++ch) {
                    a[ch] += 2 * POPCOUNT32(~(b_acc[ch] ^ w[ch][word])) - 32;
                }
            } else {
                unsigned int mask = (1u << bit) - 1;
                #pragma unroll(4)
                for (int ch = 0; ch < 4; ++ch) {
                    a[ch] += 2 * POPCOUNT32((~(b_acc[ch] ^ w[ch][word])) & mask) - bit;
                }
            }
        }
    }

    #pragma unroll(4)
    for (int ch = 0; ch < 4; ++ch) {
        acc_out[ch] = a[ch];
    }
}

// 1-channel fallback
static inline int bika_kernel_1ch_avx2(
    const float* __restrict__ in_patch,
    const float* __restrict__ nb_ch,
    const unsigned int* __restrict__ w_ch,
    int ckk
) {
    int a = 0;
    int i = 0;
    int word = 0;

    for (; i + 32 <= ckk; i += 32, word++) {
        __m256 x0 = _mm256_loadu_ps(in_patch + i + 0);
        __m256 x1 = _mm256_loadu_ps(in_patch + i + 8);
        __m256 x2 = _mm256_loadu_ps(in_patch + i + 16);
        __m256 x3 = _mm256_loadu_ps(in_patch + i + 24);

        unsigned int m0 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x0, _mm256_loadu_ps(nb_ch + i + 0), _CMP_GE_OQ));
        unsigned int m1 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x1, _mm256_loadu_ps(nb_ch + i + 8), _CMP_GE_OQ));
        unsigned int m2 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x2, _mm256_loadu_ps(nb_ch + i + 16), _CMP_GE_OQ));
        unsigned int m3 = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x3, _mm256_loadu_ps(nb_ch + i + 24), _CMP_GE_OQ));

        unsigned int bits32 = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);
        a += 2 * POPCOUNT32(~(bits32 ^ w_ch[word])) - 32;
    }

    if (i < ckk) {
        unsigned int b_acc = 0;
        int bit = 0;
        for (; i + 8 <= ckk; i += 8) {
            __m256 x = _mm256_loadu_ps(in_patch + i);
            unsigned int m = (unsigned int)_mm256_movemask_ps(_mm256_cmp_ps(x, _mm256_loadu_ps(nb_ch + i), _CMP_GE_OQ));
            b_acc |= (m << bit);
            bit += 8;
        }
        for (; i + 4 <= ckk; i += 4) {
            __m128 x = _mm_loadu_ps(in_patch + i);
            unsigned int m = (unsigned int)_mm_movemask_ps(_mm_cmpge_ps(x, _mm_loadu_ps(nb_ch + i)));
            b_acc |= (m << bit);
            bit += 4;
        }
        for (; i < ckk; ++i) {
            if (in_patch[i] >= nb_ch[i]) b_acc |= (1u << bit);
            bit++;
        }
        if (bit > 0) {
            if (bit == 32) {
                a += 2 * POPCOUNT32(~(b_acc ^ w_ch[word])) - 32;
            } else {
                unsigned int mask = (1u << bit) - 1;
                a += 2 * POPCOUNT32((~(b_acc ^ w_ch[word])) & mask) - bit;
            }
        }
    }
    return a;
}
#endif // BIKA_X86


// ============================================================
// V4 Main Forward Entrypoint
// ============================================================
torch::Tensor bika_conv2d_forward_cpu_v4(
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

    // Pre-negate bias contiguously
    std::vector<float> neg_bias(O * ckk);
    for (int i = 0; i < O * ckk; ++i) {
        neg_bias[i] = -b_ptr[i];
    }

    // Pre-pack weights if needed
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

    // Safe interior spatial bounds
    const int ho_start_safe = (pad_h > 0) ? (pad_h + stride_h - 1) / stride_h : 0;
    const int ho_end_safe = std::min(Ho, (H + pad_h - K) / stride_h + 1);
    const int wo_start_safe = (pad_w > 0) ? (pad_w + stride_w - 1) / stride_w : 0;
    const int wo_end_safe = std::min(Wo, (W + pad_w - K) / stride_w + 1);

    #pragma omp parallel
    {
        alignas(64) float gather_buf[ckk > 0 ? ckk : 1];

        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int b = 0; b < B; ++b) {
            for (int ho = 0; ho < Ho; ++ho) {
                const bool ho_safe = (ho >= ho_start_safe && ho < ho_end_safe);

                for (int wo = 0; wo < Wo; ++wo) {
                    const bool fully_safe = ho_safe && (wo >= wo_start_safe && wo < wo_end_safe);

                    // --- Fast Input Gathering ---
                    if (fully_safe) {
                        int elem = 0;
                        for (int c = 0; c < C; ++c) {
                            const float* in_c = in_ptr + ((int64_t)b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                const float* in_row = in_c + hi * W;
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
                            const float* in_c = in_ptr + ((int64_t)b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                if (hi >= 0 && hi < H) {
                                    const float* in_row = in_c + hi * W;
                                    for (int kw = 0; kw < K; ++kw) {
                                        const int wi = wo * stride_w - pad_w + kw;
                                        gather_buf[elem++] = (wi >= 0 && wi < W) ? in_row[wi] : 0.0f;
                                    }
                                } else {
                                    for (int kw = 0; kw < K; ++kw)
                                        gather_buf[elem++] = 0.0f;
                                }
                            }
                        }
                    }

                    // --- Channel Processing: 8 channels at a time ---
                    int o = 0;
                    #ifdef BIKA_X86
                    for (; o + 8 <= O; o += 8) {
                        const float* nb_ptrs[8] = {
                            neg_bias.data() + (o + 0) * ckk,
                            neg_bias.data() + (o + 1) * ckk,
                            neg_bias.data() + (o + 2) * ckk,
                            neg_bias.data() + (o + 3) * ckk,
                            neg_bias.data() + (o + 4) * ckk,
                            neg_bias.data() + (o + 5) * ckk,
                            neg_bias.data() + (o + 6) * ckk,
                            neg_bias.data() + (o + 7) * ckk
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
                        bika_kernel_8ch_avx2(gather_buf, nb_ptrs, w_ptrs, ckk, accs);

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
                        const float* nb_ptrs[4] = {
                            neg_bias.data() + (o + 0) * ckk,
                            neg_bias.data() + (o + 1) * ckk,
                            neg_bias.data() + (o + 2) * ckk,
                            neg_bias.data() + (o + 3) * ckk
                        };
                        const unsigned int* w_ptrs[4] = {
                            w_packed_ptr + (o + 0) * num_words,
                            w_packed_ptr + (o + 1) * num_words,
                            w_packed_ptr + (o + 2) * num_words,
                            w_packed_ptr + (o + 3) * num_words
                        };

                        int accs[4];
                        bika_kernel_4ch_avx2(gather_buf, nb_ptrs, w_ptrs, ckk, accs);

                        #pragma unroll(4)
                        for (int t = 0; t < 4; ++t) {
                            float res = (float)accs[t];
                            if (scale_ptr) res = res * scale_ptr[o + t] + shift_ptr[o + t];
                            if (do_relu && res < 0.0f) res = 0.0f;
                            out_ptr[((int64_t)(b * O + o + t) * Ho + ho) * Wo + wo] = res;
                        }
                    }

                    // Remaining channels (1-3)
                    for (; o < O; ++o) {
                        int acc = bika_kernel_1ch_avx2(
                            gather_buf,
                            neg_bias.data() + o * ckk,
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
