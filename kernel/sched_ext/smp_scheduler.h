#ifndef NEUROOS_SMP_SCHEDULER_H
#define NEUROOS_SMP_SCHEDULER_H

#include <stdint.h>
#include <stdbool.h>
#include "neuroos_kernel.h"
#include "neuroos_sched.h"
#include "numa_topology.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Multi-Core SMP sched_ext Dispatch Extension (Phase 6).
 *
 * Implements:
 * 1. ops.select_cpu: Cache-aware and NUMA-affinity task CPU placement.
 * 2. ops.dispatch: Per-CPU quantized micro-inference scoring with cache warmth bonus.
 * 3. Work-Stealing: Hierarchical intra-NUMA and inter-NUMA victim queue inspection.
 * 4. Migration cost accounting: Enforces minimum steal imbalance threshold to avoid ping-pong thrashing.
 */

#define NEUROOS_SMP_STEAL_THRESHOLD 2  /* Only steal if victim queue depth >= local + 2 */

struct smp_task_desc {
    uint32_t pid;
    uint32_t prev_cpu;
    uint64_t last_run_time_us;
    int8_t   features[NEUROOS_INPUT_DIM];
};

struct smp_runqueue {
    uint32_t cpu_id;
    uint32_t queue_depth;
    struct smp_task_desc tasks[NEUROOS_MAX_CANDIDATES];
};

/*
 * Select target CPU for an incoming or waking task.
 * Returns cpu_id in [0, topo->num_cpus - 1].
 */
uint32_t neuroos_smp_select_cpu(const struct numa_topology *topo,
                                const struct cpu_affinity_state *cpu_states,
                                uint32_t prev_cpu,
                                uint64_t current_time_us,
                                uint64_t last_run_time_us);

/*
 * Hierarchical work-stealing: searches for victim CPU with runnable tasks.
 * Evaluates same NUMA node first, then remote NUMA nodes.
 * Returns victim cpu_id if found, or -1 if no CPU satisfies steal conditions.
 */
int32_t neuroos_smp_steal_victim(const struct numa_topology *topo,
                                 const struct cpu_affinity_state *cpu_states,
                                 uint32_t idle_cpu_id);

/*
 * Select best candidate on a specific CPU using quantized scoring
 * augmented with cache-affinity and NUMA locality weighting.
 */
int neuroos_smp_select_candidate(const struct smp_runqueue *rq,
                                 uint32_t cpu_id,
                                 const struct numa_topology *topo,
                                 const struct neuroos_quantized_policy *policy,
                                 uint64_t current_time_us,
                                 int32_t drift_milli_sigma,
                                 struct neuroos_decision *out_decision,
                                 uint32_t q_min_us,
                                 uint32_t q_max_us);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_SMP_SCHEDULER_H */
