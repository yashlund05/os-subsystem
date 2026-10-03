#ifndef NEUROOS_PMU_HOOK_H
#define NEUROOS_PMU_HOOK_H

#include <stdint.h>
#include "telemetry_event.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Kernel telemetry producer hook (Phase 1 Week 1 completed, hardened Phase 3).
 * Populates 16-byte task_telemetry from PMU deltas + pushes to SPSC ring.
 * Producer is wait-free; drops (counts) on full without blocking dispatch.
 */

struct neuroos_pmu_sample {
    uint32_t pid;
    uint16_t elapsed_us;
    uint16_t cache_misses_delta;
    uint16_t branch_mispred_delta;
    uint16_t mem_footprint_kb;
    uint16_t flags;
};

/* Format sample into packed 16-byte telemetry struct (saturating casts). */
void neuroos_format_telemetry(const struct neuroos_pmu_sample *sample,
                               struct task_telemetry *out_event);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_PMU_HOOK_H */
