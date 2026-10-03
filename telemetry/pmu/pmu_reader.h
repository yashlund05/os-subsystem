#ifndef NEUROOS_PMU_READER_H
#define NEUROOS_PMU_READER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct pmu_context {
    int l1_dcache_miss_fd;
    int llc_miss_fd;
    int branch_mispred_fd;
    int instr_retired_fd;
};

/* Initialize PMU counters for a specific PID, or -1 for current thread/process */
bool pmu_init(struct pmu_context *ctx, int pid);

/* Start counting */
bool pmu_start(struct pmu_context *ctx);

/* Stop counting */
bool pmu_stop(struct pmu_context *ctx);

/* Read current counter values */
bool pmu_read(struct pmu_context *ctx, uint64_t *l1_misses, uint64_t *llc_misses, uint64_t *branch_mispred, uint64_t *instr_retired);

/* Cleanup file descriptors */
void pmu_cleanup(struct pmu_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_PMU_READER_H */
