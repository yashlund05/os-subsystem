/*
 * Telemetry formatting implementation. No FP, no heap, O(1).
 */

#include "pmu_hook.h"

void neuroos_format_telemetry(const struct neuroos_pmu_sample *sample,
                               struct task_telemetry *out_event) {
    if (sample == 0 || out_event == 0) {
        return;
    }
    out_event->pid = sample->pid;
    out_event->elapsed_us = sample->elapsed_us;
    out_event->cache_misses_delta = sample->cache_misses_delta;
    out_event->branch_mispred_delta = sample->branch_mispred_delta;
    out_event->mem_footprint_kb = sample->mem_footprint_kb;
    out_event->flags = sample->flags;
    out_event->_pad = 0;
}
