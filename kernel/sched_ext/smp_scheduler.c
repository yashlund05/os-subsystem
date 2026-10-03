/*
 * Multi-Core SMP sched_ext Dispatch Extension (Phase 6).
 * Core C implementation for SMP work stealing, CPU selection, and NUMA placement.
 */

#include "smp_scheduler.h"
#include "micro_infer.h"
#include "guardrail.h"

void neuroos_numa_topology_init(struct numa_topology *topo, uint32_t num_cpus, uint32_t num_nodes) {
    uint32_t c, i, j;
    if (!topo) return;

    if (num_cpus == 0) num_cpus = 1;
    if (num_cpus > NEUROOS_MAX_CPUS) num_cpus = NEUROOS_MAX_CPUS;
    if (num_nodes == 0) num_nodes = 1;
    if (num_nodes > NEUROOS_MAX_NUMA_NODES) num_nodes = NEUROOS_MAX_NUMA_NODES;

    topo->num_cpus = num_cpus;
    topo->num_nodes = num_nodes;

    for (c = 0; c < num_cpus; c++) {
        uint32_t node = (c * num_nodes) / num_cpus;
        topo->cpu_to_node[c] = node;
    }

    for (i = 0; i < num_nodes; i++) {
        topo->nodes[i].node_id = i;
        topo->nodes[i].cpu_count = 0;
        topo->nodes[i].cpu_mask = 0;
        topo->nodes[i].total_memory_bytes = (uint64_t)16 * 1024 * 1024 * 1024; /* 16GB default */
        topo->nodes[i].free_memory_bytes = topo->nodes[i].total_memory_bytes;

        for (j = 0; j < num_nodes; j++) {
            if (i == j) {
                topo->distance_matrix[i][j] = NEUROOS_NUMA_DISTANCE_LOCAL;
            } else {
                topo->distance_matrix[i][j] = NEUROOS_NUMA_DISTANCE_REMOTE;
            }
        }
    }

    for (c = 0; c < num_cpus; c++) {
        uint32_t n = topo->cpu_to_node[c];
        topo->nodes[n].cpu_count++;
        topo->nodes[n].cpu_mask |= ((uint64_t)1 << c);
    }
}

uint8_t neuroos_cpu_distance(const struct numa_topology *topo, uint32_t cpu_a, uint32_t cpu_b) {
    uint32_t node_a, node_b;
    if (!topo || cpu_a >= topo->num_cpus || cpu_b >= topo->num_cpus) {
        return NEUROOS_NUMA_DISTANCE_REMOTE;
    }
    if (cpu_a == cpu_b) {
        return NEUROOS_NUMA_DISTANCE_LOCAL;
    }
    node_a = topo->cpu_to_node[cpu_a];
    node_b = topo->cpu_to_node[cpu_b];
    if (node_a == node_b) {
        return NEUROOS_NUMA_DISTANCE_SAME_NODE;
    }
    return topo->distance_matrix[node_a][node_b];
}

uint32_t neuroos_calculate_migration_cost(const struct numa_topology *topo,
                                          uint32_t prev_cpu,
                                          uint32_t target_cpu,
                                          uint64_t current_time_us,
                                          uint64_t last_run_time_us) {
    uint8_t dist;
    bool is_hot;
    uint32_t base_cost;

    if (prev_cpu == target_cpu) {
        return 0;
    }

    dist = neuroos_cpu_distance(topo, prev_cpu, target_cpu);
    is_hot = (current_time_us >= last_run_time_us) &&
             ((current_time_us - last_run_time_us) < NEUROOS_CACHE_HOT_THRESHOLD_US);

    base_cost = (dist <= NEUROOS_NUMA_DISTANCE_SAME_NODE) ?
                NEUROOS_MIGRATION_COST_LOCAL_US : NEUROOS_MIGRATION_COST_REMOTE_US;

    /* Cache hot tasks suffer higher migration latency due to cache invalidation */
    if (is_hot) {
        base_cost *= 2;
    }
    return base_cost;
}

uint32_t neuroos_smp_select_cpu(const struct numa_topology *topo,
                                const struct cpu_affinity_state *cpu_states,
                                uint32_t prev_cpu,
                                uint64_t current_time_us,
                                uint64_t last_run_time_us) {
    uint32_t c;
    uint32_t best_cpu = prev_cpu < topo->num_cpus ? prev_cpu : 0;
    int32_t lowest_load = 0x7FFFFFFF;
    bool prev_is_hot;

    if (!topo || !cpu_states || topo->num_cpus == 0) {
        return 0;
    }

    prev_is_hot = (prev_cpu < topo->num_cpus) &&
                  (current_time_us >= last_run_time_us) &&
                  ((current_time_us - last_run_time_us) < NEUROOS_CACHE_HOT_THRESHOLD_US);

    /* If previous CPU is idle or nearly idle and cache is hot, stick to prev_cpu */
    if (prev_cpu < topo->num_cpus && prev_is_hot && cpu_states[prev_cpu].queue_depth == 0) {
        return prev_cpu;
    }

    /* Iterate over all CPUs to find least-loaded CPU, penalized by NUMA distance and migration */
    for (c = 0; c < topo->num_cpus; c++) {
        uint8_t dist = neuroos_cpu_distance(topo, prev_cpu, c);
        int32_t effective_load = (int32_t)(cpu_states[c].queue_depth * 10);

        if (dist > NEUROOS_NUMA_DISTANCE_SAME_NODE) {
            effective_load += 15; /* Interconnect distance penalty */
        } else if (dist > NEUROOS_NUMA_DISTANCE_LOCAL) {
            effective_load += 5;  /* Cross-core penalty */
        }

        if (c == prev_cpu && prev_is_hot) {
            effective_load -= 10; /* Cache warmness discount */
        }

        if (effective_load < lowest_load) {
            lowest_load = effective_load;
            best_cpu = c;
        }
    }
    return best_cpu;
}

int32_t neuroos_smp_steal_victim(const struct numa_topology *topo,
                                 const struct cpu_affinity_state *cpu_states,
                                 uint32_t idle_cpu_id) {
    uint32_t c;
    int32_t best_victim = -1;
    uint32_t max_queue = 0;
    uint32_t idle_node;

    if (!topo || !cpu_states || idle_cpu_id >= topo->num_cpus) {
        return -1;
    }

    idle_node = topo->cpu_to_node[idle_cpu_id];

    /* Step 1: Search within same NUMA node first (faster interconnect, warm cache) */
    for (c = 0; c < topo->num_cpus; c++) {
        if (c == idle_cpu_id) continue;
        if (topo->cpu_to_node[c] == idle_node) {
            if (cpu_states[c].queue_depth >= NEUROOS_SMP_STEAL_THRESHOLD &&
                cpu_states[c].queue_depth > max_queue) {
                max_queue = cpu_states[c].queue_depth;
                best_victim = (int32_t)c;
            }
        }
    }

    if (best_victim >= 0) {
        return best_victim;
    }

    /* Step 2: Search across remote NUMA nodes if intra-node steal found no victim */
    for (c = 0; c < topo->num_cpus; c++) {
        if (c == idle_cpu_id) continue;
        if (topo->cpu_to_node[c] != idle_node) {
            if (cpu_states[c].queue_depth >= (NEUROOS_SMP_STEAL_THRESHOLD + 1) &&
                cpu_states[c].queue_depth > max_queue) {
                max_queue = cpu_states[c].queue_depth;
                best_victim = (int32_t)c;
            }
        }
    }

    return best_victim;
}

int neuroos_smp_select_candidate(const struct smp_runqueue *rq,
                                 uint32_t cpu_id,
                                 const struct numa_topology *topo,
                                 const struct neuroos_quantized_policy *policy,
                                 uint64_t current_time_us,
                                 int32_t drift_milli_sigma,
                                 struct neuroos_decision *out_decision,
                                 uint32_t q_min_us,
                                 uint32_t q_max_us) {
    uint32_t i;
    int best_idx = -1;
    int32_t best_score = -0x7FFFFFFF;
    uint32_t n;

    if (!rq || !topo || !policy || !out_decision) return -1;
    if (rq->queue_depth == 0) return -1;

    n = rq->queue_depth > NEUROOS_MAX_CANDIDATES ? NEUROOS_MAX_CANDIDATES : rq->queue_depth;

    /* Guardrail check */
    if (!neuroos_guardrail_check(rq->queue_depth, drift_milli_sigma)) {
        out_decision->selected_pid = 0;
        out_decision->quantum_us = q_min_us;
        out_decision->fallback_engaged = true;
        out_decision->reason_code = (rq->queue_depth > (uint32_t)NEUROOS_MAX_QUEUE_DEPTH) ? 1u : 2u;
        return -1;
    }

    for (i = 0; i < n; i++) {
        int32_t score = neuroos_micro_infer_score(rq->tasks[i].features, policy);

        /* Apply cache affinity bonus if task ran recently on this exact CPU */
        if (rq->tasks[i].prev_cpu == cpu_id &&
            (current_time_us >= rq->tasks[i].last_run_time_us) &&
            ((current_time_us - rq->tasks[i].last_run_time_us) < NEUROOS_CACHE_HOT_THRESHOLD_US)) {
            score += 250; /* Cache warmth bonus */
        } else if (topo->cpu_to_node[rq->tasks[i].prev_cpu] != topo->cpu_to_node[cpu_id]) {
            score -= 150; /* Cross-NUMA node penalty */
        }

        if (score > best_score) {
            best_score = score;
            best_idx = (int)i;
        }
    }

    if (best_idx >= 0) {
        out_decision->selected_pid = rq->tasks[(uint32_t)best_idx].pid;
        out_decision->quantum_us = neuroos_score_to_quantum(best_score, q_min_us, q_max_us);
        out_decision->fallback_engaged = false;
        out_decision->reason_code = 0u;
    }

    return best_idx;
}
