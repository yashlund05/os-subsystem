/*
 * Online In-Kernel Drift Self-Correction Implementation (Phase 6).
 * Strictly integer, deterministic O(1), no heap, no locks, zero FP.
 */

#include "online_corrector.h"

void neuroos_online_corrector_init(struct online_drift_corrector *corr, int32_t max_clamp_bias) {
    if (!corr) return;
    corr->running_bias_offset = 0;
    corr->mean_error_ema = 0;
    corr->mean_abs_dev_ema = 500; /* Initial baseline expectation: 500 us */
    corr->sample_count = 0;
    corr->self_corrections_applied = 0;
    corr->max_clamp_bias = (max_clamp_bias > 0) ? max_clamp_bias : 20000;
}

void neuroos_online_corrector_update(struct online_drift_corrector *corr,
                                     uint32_t predicted_burst_us,
                                     uint32_t actual_burst_us) {
    int32_t raw_error;
    int32_t corrected_pred;
    int32_t residual_error;
    uint32_t abs_residual;
    int32_t alpha_q8 = NEUROOS_CORRECTOR_ALPHA_Q8;

    if (!corr) return;

    /* Raw prediction error without current correction */
    raw_error = (int32_t)actual_burst_us - (int32_t)predicted_burst_us;

    /* Residual error with current correction applied */
    corrected_pred = (int32_t)predicted_burst_us + corr->running_bias_offset;
    if (corrected_pred < 0) corrected_pred = 0;
    residual_error = (int32_t)actual_burst_us - corrected_pred;

    abs_residual = (residual_error < 0) ? (uint32_t)(-residual_error) : (uint32_t)residual_error;

    /* Update EMA of residual error: mean_ema = mean_ema + alpha * (residual_error - mean_ema) */
    corr->mean_error_ema += (int32_t)(((int64_t)alpha_q8 * (residual_error - corr->mean_error_ema)) >> 8);

    /* Update EMA of absolute deviation: dev_ema = dev_ema + alpha * (|residual| - dev_ema) */
    corr->mean_abs_dev_ema += (uint32_t)(((int64_t)alpha_q8 * ((int32_t)abs_residual - (int32_t)corr->mean_abs_dev_ema)) >> 8);
    if (corr->mean_abs_dev_ema < 10) corr->mean_abs_dev_ema = 10;

    /*
     * Stochastic Gradient / Moving Average Bias update:
     * Shift bias in direction of raw error to cancel systematic distribution shifts.
     */
    corr->running_bias_offset += (int32_t)(((int64_t)alpha_q8 * (raw_error - corr->running_bias_offset)) >> 8);

    /* Enforce safety clamps */
    if (corr->running_bias_offset > corr->max_clamp_bias) {
        corr->running_bias_offset = corr->max_clamp_bias;
    } else if (corr->running_bias_offset < -corr->max_clamp_bias) {
        corr->running_bias_offset = -corr->max_clamp_bias;
    }

    corr->sample_count++;
}

int32_t neuroos_online_corrector_adjust_score(const struct online_drift_corrector *corr,
                                              int32_t raw_score) {
    int32_t adjusted;
    if (!corr) return raw_score;

    /*
     * Map bias offset to priority score adjustment:
     * Positive bias (tasks taking longer than expected) -> boost dynamic quantum
     * or adjust priority to avoid premature preemption.
     */
    adjusted = raw_score - (corr->running_bias_offset / 10);
    return adjusted;
}

int32_t neuroos_online_corrector_get_drift_milli_sigma(const struct online_drift_corrector *corr) {
    int32_t abs_err;
    uint32_t dev;
    if (!corr || corr->sample_count < 4) {
        return 0;
    }
    abs_err = (corr->mean_error_ema < 0) ? -corr->mean_error_ema : corr->mean_error_ema;
    dev = corr->mean_abs_dev_ema > 0 ? corr->mean_abs_dev_ema : 1;

    /* Return drift in milli-sigma: (|e| * 1000) / dev */
    return (int32_t)(((uint64_t)abs_err * 1000) / dev);
}
