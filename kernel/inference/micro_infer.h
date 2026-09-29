#ifndef NEUROOS_MICRO_INFER_H
#define NEUROOS_MICRO_INFER_H

#include <stdint.h>
#include "neuroos_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NeuroOS-Lite quantized micro-inference core (Phase 2, Week 4).
 *
 * Topology: 16 -> 8 -> 1, integer only, O(1), no heap, no FP.
 * Fixed-point chain (see ml/quantization/quantize.py):
 *   x_q in int8 (host quantized with s_x), w in int8, b1 in int16 units of s_x*s_w1,
 *   accumulators int32. Output int32 score in units of s_x*s_w1*s_w2 (+b2).
 *
 * Assumptions documented per docs/Rules.md Rule 10:
 * - x86_64 with AVX2 optional; portable scalar fallback always available.
 * - TSC invariant for rdtsc_ordered timing; fallback to clock() elsewhere.
 */

#define NEUROOS_QMIN_US_DEFAULT 1000
#define NEUROOS_QMAX_US_DEFAULT 50000

/* Score N quantized int8 features with given policy. Returns int32 priority score. */
int32_t neuroos_micro_infer_score(const int8_t features[NEUROOS_INPUT_DIM],
                                   const struct neuroos_quantized_policy *policy);

/* Map int32 score to dynamic quantum in [q_min_us, q_max_us] via branchless linear LUT. */
uint32_t neuroos_score_to_quantum(int32_t score, uint32_t q_min_us, uint32_t q_max_us);

/* Ordered TSC read for overhead benchmarking (docs/Experimental-Protocol.md). */
uint64_t neuroos_rdtsc_ordered(void);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_MICRO_INFER_H */
