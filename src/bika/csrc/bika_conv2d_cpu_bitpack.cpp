// bika_conv2d_cpu_bitpack.cpp
// Int8 Movemask + Hardware POPCNT CPU Kernel for BiKA Binary Convolution
//
// Flow:
//   0. Pre-quantize ALL output channel thresholds to int8 ONCE (not per pixel!)
//   1. Per pixel: Gather input patch → quantize to int8 ONCE
//   2. Per output channel: cmpgt_epi8 → movemask → XOR → POPCNT (5 iters for CKK=144)
//
// Compile: -O3 -mavx2 -mpopcnt -fopenmp -ffast-math

#include <torch/extension.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <omp.h>

torch::Tensor bika_conv2d_forward_cpu_bitpack(
    torch::Tensor input,          // (N, C_in, H, W) float32
    torch::Tensor packed_w_bits,  // (O, num_u64) int64 — bitpacked weight signs
    torch::Tensor neg_bias_flat,  // (O, CKK_padded) float32 — flattened thresholds
    torch::Tensor out_scale,      // (O,) float32 — fused BN scale
    torch::Tensor out_shift,      // (O,) float32 — fused BN shift
    bool do_relu,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int O = packed_w_bits.size(0);
    const int num_u64 = packed_w_bits.size(1);
    const int K = 3;
    const int CKK = C * K * K;
    const int num_w32 = (CKK + 31) / 32;
    const int CKK_pad32 = num_w32 * 32;

    const int OH = (H + 2 * pad_h - K) / stride_h + 1;
    const int OW = (W + 2 * pad_w - K) / stride_w + 1;
    const int HW = H * W;
    const int padding_bits = CKK_pad32 - CKK;

    auto output = torch::empty({N, O, OH, OW}, input.options());

    const float* __restrict__ inp_ptr = input.data_ptr<float>();
    const uint32_t* __restrict__ pw32 = reinterpret_cast<const uint32_t*>(packed_w_bits.data_ptr<int64_t>());
    const float* __restrict__ nb_ptr = neg_bias_flat.data_ptr<float>();
    const int nb_stride = (int)neg_bias_flat.size(1);
    const float* sc_ptr = out_scale.numel() > 0 ? out_scale.data_ptr<float>() : nullptr;
    const float* sh_ptr = out_shift.numel() > 0 ? out_shift.data_ptr<float>() : nullptr;
    float* __restrict__ out_ptr = output.data_ptr<float>();

    // ═══ Pre-quantize ALL thresholds to int8 ONCE (shared across all pixels) ═══
    // This is O(O * CKK) work done once, not O(O * CKK * OH * OW)
    int8_t* all_thr = (int8_t*)aligned_alloc(32, (size_t)O * CKK_pad32);
    for (int o = 0; o < O; o++) {
        const float* nb_o = nb_ptr + (int64_t)o * nb_stride;
        int8_t* thr_o = all_thr + (int64_t)o * CKK_pad32;
        for (int i = 0; i < CKK; i++) {
            int v = (int)roundf(nb_o[i] * 64.0f);
            thr_o[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
        }
        // Padding: set to 127 so cmpgt(127, 0) = true → NOT(sign)=1,
        // w_bit=0 → XOR=1 → counted as matching → subtract padding_bits
        for (int i = CKK; i < CKK_pad32; i++) thr_o[i] = 127;
    }

    #pragma omp parallel
    {
        // Thread-local buffers
        float*  patch  = (float*) aligned_alloc(32, CKK_pad32 * sizeof(float));
        int8_t* inp_i8 = (int8_t*)aligned_alloc(32, CKK_pad32);
        memset(patch, 0, CKK_pad32 * sizeof(float));
        memset(inp_i8, 0, CKK_pad32);

        #pragma omp for schedule(static) collapse(2)
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < OH; oh++) {
                const int ih0 = oh * stride_h - pad_h;

                for (int ow = 0; ow < OW; ow++) {
                    const int iw0 = ow * stride_w - pad_w;

                    // ═══ Step 1: Gather input patch ═══
                    const bool interior = (ih0 >= 0) && (ih0 + K <= H) &&
                                          (iw0 >= 0) && (iw0 + K <= W);
                    int idx = 0;
                    if (interior) {
                        for (int c = 0; c < C; c++) {
                            const float* cb = inp_ptr + ((int64_t)n * C + c) * HW;
                            for (int kh = 0; kh < K; kh++) {
                                const float* row = cb + (ih0 + kh) * W + iw0;
                                patch[idx]   = row[0];
                                patch[idx+1] = row[1];
                                patch[idx+2] = row[2];
                                idx += 3;
                            }
                        }
                    } else {
                        for (int c = 0; c < C; c++) {
                            const float* cb = inp_ptr + ((int64_t)n * C + c) * HW;
                            for (int kh = 0; kh < K; kh++) {
                                int ih = ih0 + kh;
                                for (int kw = 0; kw < K; kw++) {
                                    int iw = iw0 + kw;
                                    patch[idx] = ((unsigned)ih < (unsigned)H &&
                                                  (unsigned)iw < (unsigned)W)
                                                 ? cb[ih * W + iw] : 0.0f;
                                    idx++;
                                }
                            }
                        }
                    }

                    // ═══ Step 2: Quantize input to int8 ONCE (shared across channels) ═══
                    for (int i = 0; i < CKK; i++) {
                        int v = (int)roundf(patch[i] * 64.0f);
                        inp_i8[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                    }
                    // Zero padding
                    for (int i = CKK; i < CKK_pad32; i++) inp_i8[i] = 0;

                    // ═══ Step 3: Process all output channels ═══
                    for (int o = 0; o < O; o++) {
                        const int8_t* thr_o = all_thr + (int64_t)o * CKK_pad32;
                        const uint32_t* pw_o = pw32 + (int64_t)o * num_u64 * 2;

                        // ── Int8 cmpgt → movemask → XOR → POPCNT ──
                        // cmpgt(thr, inp): 0xFF where thr > inp → NOT(sign_input)
                        // movemask: 32 bits of NOT(sign)
                        // XOR w/ weight: bits set where NOT(sign)!=weight → sign==weight (MATCH)
                        int matching = 0;
                        for (int g = 0; g < num_w32; g++) {
                            __m256i vi = _mm256_loadu_si256((__m256i*)&inp_i8[g * 32]);
                            __m256i vt = _mm256_loadu_si256((__m256i*)&thr_o[g * 32]);
                            __m256i cmp = _mm256_cmpgt_epi8(vt, vi);
                            uint32_t not_sign = (uint32_t)_mm256_movemask_epi8(cmp);
                            matching += __builtin_popcount(not_sign ^ pw_o[g]);
                        }

                        // Correct for padding (padding matches get counted)
                        matching -= padding_bits;

                        // {-1,+1} dot product = 2 * matching - CKK
                        float result = (float)(2 * matching - CKK);

                        if (sc_ptr) result = result * sc_ptr[o] + sh_ptr[o];
                        if (do_relu && result < 0.0f) result = 0.0f;

                        out_ptr[((int64_t)(n * O + o) * OH + oh) * OW + ow] = result;
                    }
                }
            }
        }

        free(patch);
        free(inp_i8);
    }

    free(all_thr);
    return output;
}
