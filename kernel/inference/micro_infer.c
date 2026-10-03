/*
 * NeuroOS-Lite quantized micro-inference core implementation.
 * Phase 2 Week 4. Strictly integer, deterministic O(1), no malloc, no FP, no locks.
 */

#include "micro_infer.h"

#include <time.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

int32_t neuroos_micro_infer_score(const int8_t features[NEUROOS_INPUT_DIM],
                                   const struct neuroos_quantized_policy *policy) {
    int32_t hidden[NEUROOS_HIDDEN_DIM];
    int i, j;

    /* Layer 1: h_j = ReLU(sum_i x_i * w1[i*8+j] + b1[j]) */
    for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) {
        int32_t acc = (int32_t)policy->b1[j];
        for (i = 0; i < NEUROOS_INPUT_DIM; i++) {
            acc += (int32_t)features[i] * (int32_t)policy->w1[j * NEUROOS_INPUT_DIM + i];
        }
        /* Integer ReLU, branchless */
        hidden[j] = acc > 0 ? acc : 0;
    }

    /* Layer 2: out = sum_j h_j * w2[j] + b2[0] */
    {
        int32_t out = policy->b2[0];
        for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) {
            out += hidden[j] * (int32_t)policy->w2[j];
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
#if defined(__x86_64__) || defined(__i386__)
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
