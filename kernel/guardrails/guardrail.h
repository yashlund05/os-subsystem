#ifndef NEUROOS_GUARDRAIL_H
#define NEUROOS_GUARDRAIL_H

#include <stdint.h>
#include <stdbool.h>
#include "neuroos_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Deterministic fail-safe guardrail engine (Phase 3, Week 5).
 * O(1), no heap, no FP, no locks. Mirrors docs/TRD.md section 6 + docs/Rules.md Rule 7.
 */

#define NEUROOS_REASON_NEURAL 0u
#define NEUROOS_REASON_QUEUE_SATURATION 1u
#define NEUROOS_REASON_DRIFT_TRIP 2u

struct neuroos_guardrail_state {
    uint32_t queue_depth;
    int32_t drift_milli_sigma;
    uint16_t reason_code;
    bool fallback_engaged;
};

/* Evaluate guardrails; returns true if neural path allowed, false if fallback required. */
bool neuroos_guardrail_eval(uint32_t queue_depth,
                             int32_t drift_milli_sigma,
                             struct neuroos_guardrail_state *out_state);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_GUARDRAIL_H */
