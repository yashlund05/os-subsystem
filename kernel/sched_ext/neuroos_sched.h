#ifndef NEUROOS_SCHED_H
#define NEUROOS_SCHED_H

#include <stdint.h>
#include <stdbool.h>
#include "neuroos_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Linux sched_ext dispatch integration (Phase 3, Week 5).
 * Portable C core callable from BPF ops.select_cpu/ops.enqueue/ops.dispatch
 * and from the Python simulator mirror (schedulers/neuroos_lite).
 *
 * Assumptions (docs/Rules.md Rule 10): Linux >= 6.12, CONFIG_BPF_SCHED=y for
 * production; simulator fallback otherwise. No FP, no malloc, bounded loops.
 */

#define NEUROOS_MAX_CANDIDATES 16

struct neuroos_ready_entry {
    uint32_t pid;
    int8_t features[NEUROOS_INPUT_DIM];
};

/* Select next task index (0..num_candidates-1) by quantized scoring.
 * Returns -1 if guardrail trips (caller must use MLFQ/SRTF fallback).
 */
int neuroos_select_candidate(const struct neuroos_ready_entry *entries,
                              uint32_t num_candidates,
                              const struct neuroos_quantized_policy *policy,
                              uint32_t queue_depth,
                              int32_t drift_milli_sigma,
                              struct neuroos_decision *out_decision,
                              uint32_t q_min_us,
                              uint32_t q_max_us);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_SCHED_H */
