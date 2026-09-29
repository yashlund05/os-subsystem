/*
 * sched_ext dispatch core (Phase 3, Week 5).
 * O(K) over K<=16 candidates, O(1) guardrail, integer-only.
 */

#include "neuroos_sched.h"
#include "neuroos_kernel.h"

/* Forward declaration to avoid cross-directory include coupling. */
int32_t neuroos_micro_infer_score(const int8_t features[NEUROOS_INPUT_DIM],
                                   const struct neuroos_quantized_policy *policy);
uint32_t neuroos_score_to_quantum(int32_t score, uint32_t q_min_us, uint32_t q_max_us);
bool neuroos_guardrail_check(uint32_t queue_depth, int32_t running_drift_milli_sigma);

int neuroos_select_candidate(const struct neuroos_ready_entry *entries,
                              uint32_t num_candidates,
                              const struct neuroos_quantized_policy *policy,
                              uint32_t queue_depth,
                              int32_t drift_milli_sigma,
                              struct neuroos_decision *out_decision,
                              uint32_t q_min_us,
                              uint32_t q_max_us) {
    uint32_t i;
    int best_idx = -1;
    int32_t best_score;
    uint32_t n;

    if (entries == 0 || policy == 0 || out_decision == 0) {
        return -1;
    }
    n = num_candidates > (uint32_t)NEUROOS_MAX_CANDIDATES ? (uint32_t)NEUROOS_MAX_CANDIDATES
                                                           : num_candidates;
    if (n == 0) {
        return -1;
    }

    /* Guardrail first: bounded O(1) fallback per TRD section 6. */
    if (!neuroos_guardrail_check(queue_depth, drift_milli_sigma)) {
        out_decision->selected_pid = 0;
        out_decision->quantum_us = q_min_us;
        out_decision->fallback_engaged = true;
        out_decision->reason_code = (queue_depth > (uint32_t)NEUROOS_MAX_QUEUE_DEPTH) ? 1u : 2u;
        return -1;
    }

    /* Argmax over quantized scores (lower score = higher priority per Python mirror?
     * C core uses max-score-wins; Python mirror maps accordingly. Documented here.) */
    best_idx = 0;
    best_score = neuroos_micro_infer_score(entries[0].features, policy);
    for (i = 1; i < n; i++) {
        int32_t s = neuroos_micro_infer_score(entries[i].features, policy);
        if (s > best_score) {
            best_score = s;
            best_idx = (int)i;
        }
    }

    out_decision->selected_pid = entries[(uint32_t)best_idx].pid;
    out_decision->quantum_us = neuroos_score_to_quantum(best_score, q_min_us, q_max_us);
    out_decision->fallback_engaged = false;
    out_decision->reason_code = 0u;
    return best_idx;
}
