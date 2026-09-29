/*
 * Guardrail engine implementation (Phase 3, Week 5).
 * Deterministic O(1) fallback per docs/Rules.md Rule 7.
 */

#include "guardrail.h"

bool neuroos_guardrail_check(uint32_t queue_depth, int32_t running_drift_milli_sigma) {
    if (queue_depth > (uint32_t)NEUROOS_MAX_QUEUE_DEPTH) {
        return false;
    }
    if (running_drift_milli_sigma > (int32_t)NEUROOS_DRIFT_THRESHOLD) {
        return false;
    }
    return true;
}

bool neuroos_guardrail_eval(uint32_t queue_depth,
                             int32_t drift_milli_sigma,
                             struct neuroos_guardrail_state *out_state) {
    bool pass = neuroos_guardrail_check(queue_depth, drift_milli_sigma);
    if (out_state != 0) {
        out_state->queue_depth = queue_depth;
        out_state->drift_milli_sigma = drift_milli_sigma;
        out_state->fallback_engaged = pass ? false : true;
        if (pass) {
            out_state->reason_code = NEUROOS_REASON_NEURAL;
        } else if (queue_depth > (uint32_t)NEUROOS_MAX_QUEUE_DEPTH) {
            out_state->reason_code = NEUROOS_REASON_QUEUE_SATURATION;
        } else {
            out_state->reason_code = NEUROOS_REASON_DRIFT_TRIP;
        }
    }
    return pass;
}
