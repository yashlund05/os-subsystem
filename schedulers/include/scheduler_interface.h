#ifndef NEUROOS_SCHEDULER_INTERFACE_H
#define NEUROOS_SCHEDULER_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Task state descriptor for simulator and kernel abstractions */
struct task_descriptor {
    uint32_t pid;
    uint64_t arrival_time_us;
    uint64_t total_burst_us;
    uint64_t executed_burst_us;
    uint64_t remaining_burst_us;
    uint64_t last_dispatch_time_us;
    uint64_t waiting_time_us;
    uint32_t context_switches;
    uint32_t priority_level;
    void *priv;
};

/* Scheduling algorithm enumeration */
typedef enum {
    SCHED_ALGO_FCFS = 0,
    SCHED_ALGO_SJF,
    SCHED_ALGO_SRTF,
    SCHED_ALGO_ROUND_ROBIN,
    SCHED_ALGO_MLFQ,
    SCHED_ALGO_NEUROOS_LITE
} sched_algo_type_t;

/* Scheduling metrics summary */
struct sched_metrics {
    uint64_t total_tasks_completed;
    uint64_t total_turnaround_time_us;
    uint64_t total_waiting_time_us;
    uint64_t total_response_time_us;
    uint64_t total_context_switches;
    uint64_t total_inference_time_ns;
    uint32_t fallback_count;
};

/* Unified Scheduler Operations Interface */
struct scheduler_ops {
    const char *name;
    sched_algo_type_t algo_type;

    int (*init)(void **state, void *config);
    int (*enqueue)(void *state, struct task_descriptor *task);
    struct task_descriptor *(*pick_next)(void *state, uint64_t current_time_us, uint32_t *quantum_us);
    int (*yield)(void *state, struct task_descriptor *task);
    int (*dequeue)(void *state, uint32_t pid);
    void (*tick)(void *state, uint64_t elapsed_us);
    void (*get_metrics)(void *state, struct sched_metrics *out_metrics);
    void (*destroy)(void *state);
};

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_SCHEDULER_INTERFACE_H */
