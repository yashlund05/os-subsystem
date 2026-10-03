#ifndef NEUROOS_NUMA_TOPOLOGY_H
#define NEUROOS_NUMA_TOPOLOGY_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NUMA and SMP Topology Definitions for NeuroOS-Lite (Phase 6).
 * Supports up to 64 CPUs organized across up to 8 NUMA nodes.
 * Designed for strict zero-allocation, bounded O(1) in-kernel execution.
 */

#define NEUROOS_MAX_CPUS        64
#define NEUROOS_MAX_NUMA_NODES   8

#define NEUROOS_NUMA_DISTANCE_LOCAL       10  /* Same CPU / same local memory */
#define NEUROOS_NUMA_DISTANCE_SAME_NODE   12  /* Same NUMA node, different core */
#define NEUROOS_NUMA_DISTANCE_REMOTE      20  /* Cross-NUMA node hop */

#define NEUROOS_CACHE_HOT_THRESHOLD_US   500  /* Cache lines considered warm within 500 us */
#define NEUROOS_MIGRATION_COST_LOCAL_US    5  /* Intra-node migration context penalty */
#define NEUROOS_MIGRATION_COST_REMOTE_US  15  /* Cross-NUMA node migration penalty */

struct numa_node_info {
    uint32_t node_id;
    uint32_t cpu_count;
    uint64_t cpu_mask;               /* Bitmask of CPUs belonging to this node */
    uint64_t total_memory_bytes;
    uint64_t free_memory_bytes;
};

struct numa_topology {
    uint32_t num_cpus;
    uint32_t num_nodes;
    uint32_t cpu_to_node[NEUROOS_MAX_CPUS];
    uint8_t  distance_matrix[NEUROOS_MAX_NUMA_NODES][NEUROOS_MAX_NUMA_NODES];
    struct numa_node_info nodes[NEUROOS_MAX_NUMA_NODES];
};

struct cpu_affinity_state {
    uint32_t cpu_id;
    uint32_t current_pid;
    uint32_t last_pid;
    uint64_t last_dispatch_time_us;
    uint32_t queue_depth;
    bool     is_idle;
};

/*
 * Initialize default symmetric NUMA topology:
 * Uniform distribution of CPUs across nodes with standard distance metrics.
 */
void neuroos_numa_topology_init(struct numa_topology *topo, uint32_t num_cpus, uint32_t num_nodes);

/* Return NUMA distance between two CPUs */
uint8_t neuroos_cpu_distance(const struct numa_topology *topo, uint32_t cpu_a, uint32_t cpu_b);

/*
 * Calculate migration penalty in microseconds based on cache warmth and NUMA distance.
 */
uint32_t neuroos_calculate_migration_cost(const struct numa_topology *topo,
                                          uint32_t prev_cpu,
                                          uint32_t target_cpu,
                                          uint64_t current_time_us,
                                          uint64_t last_run_time_us);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_NUMA_TOPOLOGY_H */
