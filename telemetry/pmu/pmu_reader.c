#include "pmu_reader.h"
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int open_pmu_event(uint32_t type, uint64_t config, int pid) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    
    return perf_event_open(&pe, pid, -1, -1, 0);
}

bool pmu_init(struct pmu_context *ctx, int pid) {
    if (!ctx) return false;
    
    ctx->l1_dcache_miss_fd = -1;
    ctx->llc_miss_fd = -1;
    ctx->branch_mispred_fd = -1;
    ctx->instr_retired_fd = -1;
    
    /* Hardware cache event configuration for L1 D-Cache misses */
    uint64_t l1d_miss = (PERF_COUNT_HW_CACHE_L1D) |
                        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                        
    ctx->l1_dcache_miss_fd = open_pmu_event(PERF_TYPE_HW_CACHE, l1d_miss, pid);
    
    /* LLC misses */
    uint64_t llc_miss = (PERF_COUNT_HW_CACHE_LL) |
                        (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                        
    ctx->llc_miss_fd = open_pmu_event(PERF_TYPE_HW_CACHE, llc_miss, pid);
    
    /* Branch mispredictions */
    ctx->branch_mispred_fd = open_pmu_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, pid);
    
    /* Instructions retired */
    ctx->instr_retired_fd = open_pmu_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, pid);
    
    /* Return true if at least one counter was successfully opened */
    return (ctx->l1_dcache_miss_fd >= 0 || ctx->llc_miss_fd >= 0 || 
            ctx->branch_mispred_fd >= 0 || ctx->instr_retired_fd >= 0);
}

bool pmu_start(struct pmu_context *ctx) {
    if (!ctx) return false;
    
    if (ctx->l1_dcache_miss_fd >= 0) ioctl(ctx->l1_dcache_miss_fd, PERF_EVENT_IOC_ENABLE, 0);
    if (ctx->llc_miss_fd >= 0) ioctl(ctx->llc_miss_fd, PERF_EVENT_IOC_ENABLE, 0);
    if (ctx->branch_mispred_fd >= 0) ioctl(ctx->branch_mispred_fd, PERF_EVENT_IOC_ENABLE, 0);
    if (ctx->instr_retired_fd >= 0) ioctl(ctx->instr_retired_fd, PERF_EVENT_IOC_ENABLE, 0);
    
    return true;
}

bool pmu_stop(struct pmu_context *ctx) {
    if (!ctx) return false;
    
    if (ctx->l1_dcache_miss_fd >= 0) ioctl(ctx->l1_dcache_miss_fd, PERF_EVENT_IOC_DISABLE, 0);
    if (ctx->llc_miss_fd >= 0) ioctl(ctx->llc_miss_fd, PERF_EVENT_IOC_DISABLE, 0);
    if (ctx->branch_mispred_fd >= 0) ioctl(ctx->branch_mispred_fd, PERF_EVENT_IOC_DISABLE, 0);
    if (ctx->instr_retired_fd >= 0) ioctl(ctx->instr_retired_fd, PERF_EVENT_IOC_DISABLE, 0);
    
    return true;
}

static uint64_t read_counter(int fd) {
    uint64_t val = 0;
    if (fd >= 0) {
        if (read(fd, &val, sizeof(uint64_t)) < 0) {
            return 0;
        }
    }
    return val;
}

bool pmu_read(struct pmu_context *ctx, uint64_t *l1_misses, uint64_t *llc_misses, uint64_t *branch_mispred, uint64_t *instr_retired) {
    if (!ctx) return false;
    
    if (l1_misses) *l1_misses = read_counter(ctx->l1_dcache_miss_fd);
    if (llc_misses) *llc_misses = read_counter(ctx->llc_miss_fd);
    if (branch_mispred) *branch_mispred = read_counter(ctx->branch_mispred_fd);
    if (instr_retired) *instr_retired = read_counter(ctx->instr_retired_fd);
    
    return true;
}

void pmu_cleanup(struct pmu_context *ctx) {
    if (!ctx) return;
    
    if (ctx->l1_dcache_miss_fd >= 0) close(ctx->l1_dcache_miss_fd);
    if (ctx->llc_miss_fd >= 0) close(ctx->llc_miss_fd);
    if (ctx->branch_mispred_fd >= 0) close(ctx->branch_mispred_fd);
    if (ctx->instr_retired_fd >= 0) close(ctx->instr_retired_fd);
    
    ctx->l1_dcache_miss_fd = -1;
    ctx->llc_miss_fd = -1;
    ctx->branch_mispred_fd = -1;
    ctx->instr_retired_fd = -1;
}
