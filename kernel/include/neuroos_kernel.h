#ifndef NEUROOS_KERNEL_H
#define NEUROOS_KERNEL_H

#include <stdint.h>
#include <stdbool.h>
#include "telemetry_event.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NeuroOS-Lite Kernel Fast-Path Interface
 *
 * Requirements from Specification:
 * - Sub-50ns target evaluation latency (strict budget: <= 45 ns target)
 * - Zero floating-point operations in fast path (pure integer/fixed-point arithmetic)
 * - Deterministic O(1) fail-safe fallback when queue depth > 1024 or drift > 3 sigma
 */

#define NEUROOS_INPUT_DIM       16
#define NEUROOS_HIDDEN_DIM      8
#define NEUROOS_OUTPUT_DIM      1

#define NEUROOS_MAX_QUEUE_DEPTH 1024
#define NEUROOS_DRIFT_THRESHOLD 3000 /* 3.0 sigma in milli-sigma fixed-point */

/*
 * Quantized Integer Policy Weights (16 -> 8 -> 1 MLP)
 * Stored in read-mostly shared atomic cache.
 */
struct neuroos_quantized_policy {
    int8_t  w1[NEUROOS_INPUT_DIM * NEUROOS_HIDDEN_DIM]; /* Layer 1 weights (int8) */
    int16_t b1[NEUROOS_HIDDEN_DIM];                    /* Layer 1 bias (int16) */
    int8_t  w2[NEUROOS_HIDDEN_DIM * NEUROOS_OUTPUT_DIM]; /* Layer 2 weights (int8) */
    int32_t b2[NEUROOS_OUTPUT_DIM];                    /* Layer 2 bias (int32) */
    uint32_t version;                                  /* Monotonic policy version */
};

/*
 * Dispatch decision result
 */
struct neuroos_decision {
    uint32_t selected_pid;
    uint32_t quantum_us;
    bool fallback_engaged;
    uint16_t reason_code;
};

/*
 * Fast-path Micro-Inference Declaration
 * Computes priority score or remaining burst estimate for a task using int8 fixed-point arithmetic.
 */
int32_t neuroos_micro_infer_score(const int8_t features[NEUROOS_INPUT_DIM],
                                  const struct neuroos_quantized_policy *policy);

/*
 * Guardrail check: returns true if guardrail passes; false if fallback must trip.
 */
bool neuroos_guardrail_check(uint32_t queue_depth, int32_t running_drift_milli_sigma);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_KERNEL_H */
