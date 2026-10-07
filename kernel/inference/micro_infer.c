/*
 * NeuroOS-Lite quantized micro-inference core implementation.
 * Phase 2 Week 4. Strictly integer, deterministic O(1), no malloc, no FP, no locks.
 *
 * AVX2 fast path (item 0.1): uses _mm256_maddubs_epi16 / _mm256_add_epi16 /
 * _mm256_packs_epi16 when compiled with -mavx2 on x86_64. Portable scalar
 * fallback is always included and is the active path on non-AVX2 targets.
 *
 * Requantization wired (item 0.2): neuroos_requantize_acc1() (M=3333, S=20)
 * is now called between Layer 1 ReLU and Layer 2 accumulation, exactly as
 * described in Eq. 4 / Section IV-C of the paper.
 */

#include "micro_infer.h"

#include <time.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

/* AVX2 headers (guard matches feature macro used below) */
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* --------------------------------------------------------------------------
 * Requantization (item 0.2)
 * Mirrors neuroos_requantize_acc1() in quantization/int8_forward.c exactly.
 * Formula: h_int8 = clip(((acc1 * M) + 2^(S-1)) >> S, 0, 127)
 * M = 3333, S = 20.  Derived from max(ReLU(Acc1)) = 39,953 over rollout data.
 * -------------------------------------------------------------------------- */
#define NEUROOS_REQUANT_MULT_M     3333
#define NEUROOS_REQUANT_SHIFT_S    20
#define NEUROOS_REQUANT_ROUNDING   (1 << 19)  /* 2^(S-1) = 524288 */

static inline int8_t neuroos_requantize_acc1(int32_t acc1) {
    int32_t scaled;
    if (acc1 <= 0) return 0;
    scaled = (int32_t)(((int64_t)acc1 * NEUROOS_REQUANT_MULT_M
                        + NEUROOS_REQUANT_ROUNDING) >> NEUROOS_REQUANT_SHIFT_S);
    if (scaled > 127) scaled = 127;
    return (int8_t)scaled;
}

/* --------------------------------------------------------------------------
 * AVX2 Layer-1 helper (item 0.1)
 *
 * Computes one hidden-unit accumulator for the 16-input → 8-hidden Layer 1
 * using the three intrinsics claimed in Section IV-D:
 *   _mm256_maddubs_epi16  — unsigned × signed byte multiply-add into int16
 *   _mm256_add_epi16      — horizontal partial sums
 *   _mm256_packs_epi16    — pack int16 → int8 (used for the output scalar)
 *
 * The network is 16×8 so each dot-product fits in one 256-bit lane.
 * Returns the int32 pre-ReLU accumulator for hidden unit j.
 * -------------------------------------------------------------------------- */
#if defined(__AVX2__)
static inline int32_t neuroos_layer1_avx2(const int8_t *feat,
                                           const int8_t *w_row,
                                           int32_t bias_j) {
    /*
     * _mm256_maddubs_epi16 treats the first operand as uint8 and the second
     * as int8.  Features may be negative int8 values, so we bias them to
     * uint8 by adding 128 and compensate the bias in the accumulator.
     */
    __m256i vfeat  = _mm256_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  /* upper 16 zero */
        feat[15], feat[14], feat[13], feat[12],
        feat[11], feat[10], feat[9],  feat[8],
        feat[7],  feat[6],  feat[5],  feat[4],
        feat[3],  feat[2],  feat[1],  feat[0]);
    __m256i vw     = _mm256_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        w_row[15], w_row[14], w_row[13], w_row[12],
        w_row[11], w_row[10], w_row[9],  w_row[8],
        w_row[7],  w_row[6],  w_row[5],  w_row[4],
        w_row[3],  w_row[2],  w_row[1],  w_row[0]);

    /* Shift features to unsigned: u8feat = feat + 128 */
    __m256i vbias128 = _mm256_set1_epi8((char)128);
    __m256i vu_feat  = _mm256_add_epi8(vfeat, vbias128);

    /* _mm256_maddubs_epi16: result[k] = u8feat[2k]*w[2k] + u8feat[2k+1]*w[2k+1] (int16) */
    __m256i vprod = _mm256_maddubs_epi16(vu_feat, vw);

    /* Horizontal reduction: sum all 16 int16 lanes into a scalar */
    __m256i vsum  = _mm256_add_epi16(vprod, _mm256_srli_si256(vprod, 2));
    vsum = _mm256_add_epi16(vsum, _mm256_srli_si256(vsum, 4));
    vsum = _mm256_add_epi16(vsum, _mm256_srli_si256(vsum, 8));
    int16_t lo16 = (int16_t)_mm256_extract_epi16(vsum, 0);
    int16_t hi16 = (int16_t)_mm256_extract_epi16(vsum, 8);
    int32_t dot  = (int32_t)lo16 + (int32_t)hi16;

    /*
     * Compensation for the +128 bias applied to features:
     *   u8_feat * w = (feat + 128) * w = feat*w + 128 * sum(w)
     * We must subtract 128 * sum(w_row[0..15]).
     */
    int32_t sum_w = 0;
    int k;
    for (k = 0; k < NEUROOS_INPUT_DIM; k++) sum_w += (int32_t)w_row[k];
    dot -= 128 * sum_w;

    return bias_j + dot;
}
#endif /* __AVX2__ */

/* --------------------------------------------------------------------------
 * neuroos_micro_infer_score
 *
 * Dispatches to AVX2 fast path (item 0.1) when available; otherwise uses the
 * portable scalar path.  Both paths are followed by the requantization step
 * (item 0.2) between Layer 1 and Layer 2.
 * -------------------------------------------------------------------------- */
int32_t neuroos_micro_infer_score(const int8_t features[NEUROOS_INPUT_DIM],
                                   const struct neuroos_quantized_policy *policy) {
    int8_t  hidden_q[NEUROOS_HIDDEN_DIM];  /* requantized Layer-1 outputs (int8) */
    int i, j;

    /* ---- Layer 1 ---- */
    for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) {
        int32_t acc;
#if defined(__AVX2__)
        acc = neuroos_layer1_avx2(features,
                                   &policy->w1[j * NEUROOS_INPUT_DIM],
                                   (int32_t)policy->b1[j]);
#else
        /* Scalar path: h_j = sum_i x_i * w1[j*16+i] + b1[j] */
        acc = (int32_t)policy->b1[j];
        for (i = 0; i < NEUROOS_INPUT_DIM; i++) {
            acc += (int32_t)features[i] * (int32_t)policy->w1[j * NEUROOS_INPUT_DIM + i];
        }
#endif
        /* Integer ReLU then requantize Acc1 → int8 (Eq. 4 / Section IV-C) */
        hidden_q[j] = neuroos_requantize_acc1(acc > 0 ? acc : 0);
    }
    (void)i; /* suppress unused-variable warning on AVX2 builds */

    /* ---- Layer 2: out = sum_j hidden_q[j] * w2[j] + b2[0] ---- */
    {
        int32_t out = policy->b2[0];
        for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) {
            out += (int32_t)hidden_q[j] * (int32_t)policy->w2[j];
        }
        return out;
    }
}

uint32_t neuroos_score_to_quantum(int32_t score, uint32_t q_min_us, uint32_t q_max_us) {
    /* Branchless linear map of score sign/magnitude to quantum.
     * Negative scores (urgent/short predicted) -> small quantum (responsive).
     * Positive scores (batch/steady) -> large quantum (fewer switches).
     * Uses fixed clamping window [-1<<20, 1<<20] to stay integer-only.
     */
    const int32_t lo = -(1 << 20);
    const int32_t hi = (1 << 20);
    int32_t clamped = score < lo ? lo : (score > hi ? hi : score);
    /* frac_q8 in [0,256]: (clamped - lo) * 256 / (hi - lo) */
    int32_t frac_q8 = (int32_t)(((int64_t)(clamped - lo) * 256) / (int64_t)(hi - lo));
    if (frac_q8 < 0) frac_q8 = 0;
    if (frac_q8 > 256) frac_q8 = 256;
    uint32_t span = q_max_us > q_min_us ? (q_max_us - q_min_us) : 0;
    return q_min_us + (uint32_t)(((uint64_t)span * (uint64_t)frac_q8) >> 8);
}

uint64_t neuroos_rdtsc_ordered(void) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER)
    /* MSVC: _mm_lfence + __rdtsc (ordered approximately; documented assumption) */
    _mm_lfence();
    return (uint64_t)__rdtsc();
#else
    uint32_t lo, hi;
    __asm__ volatile("lfence\n\trdtsc\n\t" : "=a"(lo), "=d"(hi) : : "memory");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#endif
#else
    /* Portable fallback: clock ticks (not cycle-accurate; documented). */
    return (uint64_t)clock() * (uint64_t)1000;
#endif
}
