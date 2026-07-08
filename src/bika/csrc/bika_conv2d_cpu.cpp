#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>
#include <algorithm>
#include <cstring>

// ============================================================
// Platform-specific SIMD + popcount includes
// ============================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define BIKA_X86 1
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define BIKA_ARM64 1
    #include <arm_neon.h>
#endif

#ifdef _MSC_VER
    #include <intrin.h>
    #define POPCOUNT32(x) __popcnt(x)
    #define POPCOUNT64(x) __popcnt64(x)
#else
    #define POPCOUNT32(x) __builtin_popcount(x)
    #define POPCOUNT64(x) __builtin_popcountll(x)
#endif

// ============================================================
// Fused binarize + XNOR + popcount with 64-bit accumulation.
// Processes input against bias and weight, returns BNN dot product.
// Uses POPCOUNT64 to halve the number of popcount operations.
// ============================================================
#ifdef BIKA_X86

static inline int fused_binarize_xnor_popcnt_avx2(
    const float* __restrict__ input_vals,
    const float* __restrict__ neg_bias_vals,
    const unsigned int* __restrict__ w_packed,
    int ckk
) {
    int acc = 0;
    unsigned long long x_acc64 = 0;  // accumulate 64 bits before popcount
    int bit_pos = 0;
    int word_idx = 0;
    int i = 0;

    for (; i + 8 <= ckk; i += 8) {
        __m256 x = _mm256_loadu_ps(input_vals + i);
        __m256 nb = _mm256_loadu_ps(neg_bias_vals + i);
        __m256 cmp = _mm256_cmp_ps(x, nb, _CMP_GE_OQ);
        unsigned int mask8 = (unsigned int)_mm256_movemask_ps(cmp);

        x_acc64 |= ((unsigned long long)mask8 << bit_pos);
        bit_pos += 8;

        // Flush 64 bits: do XNOR with 2 weight words and a single POPCOUNT64
        if (bit_pos >= 64) {
            unsigned long long w64;
            memcpy(&w64, w_packed + word_idx, sizeof(unsigned long long));
            unsigned long long xnor_val = ~(x_acc64 ^ w64);
            acc += 2 * (int)POPCOUNT64(xnor_val) - 64;
            word_idx += 2;

            int overflow = bit_pos - 64;
            if (overflow > 0) {
                x_acc64 = (unsigned long long)mask8 >> (8 - overflow);
            } else {
                x_acc64 = 0;
            }
            bit_pos = overflow;
        }
    }

    // SSE tail (4 at a time)
    for (; i + 4 <= ckk; i += 4) {
        __m128 x = _mm_loadu_ps(input_vals + i);
        __m128 nb = _mm_loadu_ps(neg_bias_vals + i);
        __m128 cmp = _mm_cmpge_ps(x, nb);
        unsigned int mask4 = (unsigned int)_mm_movemask_ps(cmp);

        x_acc64 |= ((unsigned long long)mask4 << bit_pos);
        bit_pos += 4;

        if (bit_pos >= 64) {
            unsigned long long w64;
            memcpy(&w64, w_packed + word_idx, sizeof(unsigned long long));
            unsigned long long xnor_val = ~(x_acc64 ^ w64);
            acc += 2 * (int)POPCOUNT64(xnor_val) - 64;
            word_idx += 2;
            int overflow = bit_pos - 64;
            if (overflow > 0) {
                x_acc64 = (unsigned long long)mask4 >> (4 - overflow);
            } else {
                x_acc64 = 0;
            }
            bit_pos = overflow;
        }
    }

    // Scalar tail
    for (; i < ckk; i++) {
        if (input_vals[i] >= neg_bias_vals[i]) {
            x_acc64 |= (1ULL << bit_pos);
        }
        bit_pos++;
        if (bit_pos == 64) {
            unsigned long long w64;
            memcpy(&w64, w_packed + word_idx, sizeof(unsigned long long));
            unsigned long long xnor_val = ~(x_acc64 ^ w64);
            acc += 2 * (int)POPCOUNT64(xnor_val) - 64;
            word_idx += 2;
            x_acc64 = 0;
            bit_pos = 0;
        }
    }

    // Flush remaining bits (could be up to 63 bits)
    if (bit_pos > 0) {
        // Handle remaining 32-bit words
        while (bit_pos >= 32) {
            unsigned int x32 = (unsigned int)(x_acc64 & 0xFFFFFFFFULL);
            unsigned int xnor_val = ~(x32 ^ w_packed[word_idx]);
            acc += 2 * POPCOUNT32(xnor_val) - 32;
            word_idx++;
            x_acc64 >>= 32;
            bit_pos -= 32;
        }
        if (bit_pos > 0) {
            unsigned int mask = (1u << bit_pos) - 1;
            unsigned int x32 = (unsigned int)(x_acc64 & mask);
            unsigned int xnor_val = (~(x32 ^ w_packed[word_idx])) & mask;
            acc += 2 * POPCOUNT32(xnor_val) - bit_pos;
        }
    }

    return acc;
}

// Dual-channel variant: processes TWO output channels simultaneously,
// sharing the input load across both. Reduces L1 bandwidth by 2x.
static inline void fused_dual_channel_avx2(
    const float* __restrict__ input_vals,
    const float* __restrict__ neg_bias_0,
    const float* __restrict__ neg_bias_1,
    const unsigned int* __restrict__ w_packed_0,
    const unsigned int* __restrict__ w_packed_1,
    int ckk,
    int* __restrict__ acc_out_0,
    int* __restrict__ acc_out_1
) {
    int acc0 = 0, acc1 = 0;
    unsigned int x_acc0 = 0, x_acc1 = 0;
    int bit_pos = 0;
    int word_idx = 0;
    int i = 0;

    for (; i + 8 <= ckk; i += 8) {
        // Load input ONCE, reuse for both channels
        __m256 x = _mm256_loadu_ps(input_vals + i);

        // Channel 0
        __m256 nb0 = _mm256_loadu_ps(neg_bias_0 + i);
        __m256 cmp0 = _mm256_cmp_ps(x, nb0, _CMP_GE_OQ);
        unsigned int mask0 = (unsigned int)_mm256_movemask_ps(cmp0);

        // Channel 1
        __m256 nb1 = _mm256_loadu_ps(neg_bias_1 + i);
        __m256 cmp1 = _mm256_cmp_ps(x, nb1, _CMP_GE_OQ);
        unsigned int mask1 = (unsigned int)_mm256_movemask_ps(cmp1);

        x_acc0 |= (mask0 << bit_pos);
        x_acc1 |= (mask1 << bit_pos);
        bit_pos += 8;

        if (bit_pos >= 32) {
            unsigned int xnor0 = ~(x_acc0 ^ w_packed_0[word_idx]);
            unsigned int xnor1 = ~(x_acc1 ^ w_packed_1[word_idx]);
            acc0 += 2 * POPCOUNT32(xnor0) - 32;
            acc1 += 2 * POPCOUNT32(xnor1) - 32;
            word_idx++;

            int overflow = bit_pos - 32;
            if (overflow > 0) {
                x_acc0 = mask0 >> (8 - overflow);
                x_acc1 = mask1 >> (8 - overflow);
            } else {
                x_acc0 = 0;
                x_acc1 = 0;
            }
            bit_pos = overflow;
        }
    }

    // Scalar tail
    for (; i < ckk; i++) {
        unsigned int bit = (input_vals[i] >= neg_bias_0[i]) ? 1u : 0u;
        x_acc0 |= (bit << bit_pos);
        bit = (input_vals[i] >= neg_bias_1[i]) ? 1u : 0u;
        x_acc1 |= (bit << bit_pos);
        bit_pos++;
        if (bit_pos == 32) {
            unsigned int xnor0 = ~(x_acc0 ^ w_packed_0[word_idx]);
            unsigned int xnor1 = ~(x_acc1 ^ w_packed_1[word_idx]);
            acc0 += 2 * POPCOUNT32(xnor0) - 32;
            acc1 += 2 * POPCOUNT32(xnor1) - 32;
            word_idx++;
            x_acc0 = 0; x_acc1 = 0;
            bit_pos = 0;
        }
    }

    if (bit_pos > 0) {
        unsigned int mask = (1u << bit_pos) - 1;
        unsigned int xnor0 = (~(x_acc0 ^ w_packed_0[word_idx])) & mask;
        unsigned int xnor1 = (~(x_acc1 ^ w_packed_1[word_idx])) & mask;
        acc0 += 2 * POPCOUNT32(xnor0) - bit_pos;
        acc1 += 2 * POPCOUNT32(xnor1) - bit_pos;
    }

    *acc_out_0 = acc0;
    *acc_out_1 = acc1;
}

#else
// Scalar fallback for single channel
static inline int fused_binarize_xnor_popcnt_scalar(
    const float* __restrict__ input_vals,
    const float* __restrict__ neg_bias_vals,
    const unsigned int* __restrict__ w_packed,
    int ckk
) {
    int acc = 0;
    unsigned int x_accumulator = 0;
    int bit_pos = 0;
    int word_idx = 0;
    for (int i = 0; i < ckk; i++) {
        if (input_vals[i] >= neg_bias_vals[i]) {
            x_accumulator |= (1u << bit_pos);
        }
        bit_pos++;
        if (bit_pos == 32) {
            unsigned int xnor_val = ~(x_accumulator ^ w_packed[word_idx]);
            acc += 2 * POPCOUNT32(xnor_val) - 32;
            word_idx++;
            x_accumulator = 0;
            bit_pos = 0;
        }
    }
    if (bit_pos > 0) {
        unsigned int mask = (1u << bit_pos) - 1;
        unsigned int xnor_val = (~(x_accumulator ^ w_packed[word_idx])) & mask;
        acc += 2 * POPCOUNT32(xnor_val) - bit_pos;
    }
    return acc;
}
#endif

torch::Tensor bika_conv2d_forward_cpu(
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

    // Interior region bounds (no padding needed)
    const int ho_start_safe = (pad_h + stride_h - 1) / stride_h;
    const int ho_end_safe = std::min(Ho, (H + pad_h - K) / stride_h + 1);
    const int wo_start_safe = (pad_w + stride_w - 1) / stride_w;
    const int wo_end_safe = std::min(Wo, (W + pad_w - K) / stride_w + 1);

    #pragma omp parallel
    {
        // Per-thread aligned gather buffer
        alignas(32) float gather_buf[ckk > 0 ? ckk : 1];

        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int b = 0; b < B; ++b) {
            for (int ho = 0; ho < Ho; ++ho) {
                const bool ho_safe = (ho >= ho_start_safe && ho < ho_end_safe);

                for (int wo = 0; wo < Wo; ++wo) {
                    const bool fully_safe = ho_safe && (wo >= wo_start_safe && wo < wo_end_safe);

                    // ── Gather input patch ──
                    if (fully_safe) {
                        int elem = 0;
                        for (int c = 0; c < C; ++c) {
                            const float* in_c = in_ptr + (b * C + c) * H * W;
                            for (int kh = 0; kh < K; ++kh) {
                                const int hi = ho * stride_h - pad_h + kh;
                                const float* in_row = in_c + hi * W;
                                const int wi_base = wo * stride_w - pad_w;
                                if (K == 3) {
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
                            const float* in_c = in_ptr + (b * C + c) * H * W;
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

                    // ── Process output channels (dual-channel when possible) ──
                    int o = 0;
                    #ifdef BIKA_X86
                    // Process pairs of output channels to share input loads
                    for (; o + 1 < O; o += 2) {
                        const float* nb0 = neg_bias.data() + o * ckk;
                        const float* nb1 = neg_bias.data() + (o + 1) * ckk;
                        const unsigned int* w0 = w_packed_ptr + o * num_words;
                        const unsigned int* w1 = w_packed_ptr + (o + 1) * num_words;

                        // Prefetch next pair's bias
                        if (o + 3 < O) {
                            __builtin_prefetch(neg_bias.data() + (o + 2) * ckk, 0, 1);
                            __builtin_prefetch(neg_bias.data() + (o + 3) * ckk, 0, 1);
                        }

                        int acc0, acc1;
                        fused_dual_channel_avx2(
                            gather_buf, nb0, nb1, w0, w1, ckk, &acc0, &acc1
                        );

                        float res0 = (float)acc0;
                        float res1 = (float)acc1;
                        if (scale_ptr) {
                            res0 = res0 * scale_ptr[o] + shift_ptr[o];
                            res1 = res1 * scale_ptr[o+1] + shift_ptr[o+1];
                        }
                        if (do_relu) {
                            if (res0 < 0.0f) res0 = 0.0f;
                            if (res1 < 0.0f) res1 = 0.0f;
                        }

                        out_ptr[((b * O + o) * Ho + ho) * Wo + wo] = res0;
                        out_ptr[((b * O + o + 1) * Ho + ho) * Wo + wo] = res1;
                    }
                    #endif

                    // Handle odd remaining channel
                    for (; o < O; ++o) {
                        const float* neg_b_o = neg_bias.data() + o * ckk;
                        const unsigned int* w_o = w_packed_ptr + o * num_words;

                        #ifdef BIKA_X86
                        int acc = fused_binarize_xnor_popcnt_avx2(
                            gather_buf, neg_b_o, w_o, ckk
                        );
                        #else
                        int acc = fused_binarize_xnor_popcnt_scalar(
                            gather_buf, neg_b_o, w_o, ckk
                        );
                        #endif

                        float res = (float)acc;
                        if (scale_ptr) res = res * scale_ptr[o] + shift_ptr[o];
                        if (do_relu && res < 0.0f) res = 0.0f;
                        out_ptr[((b * O + o) * Ho + ho) * Wo + wo] = res;
                    }
                }
            }
        }
    }

    return output;
}
