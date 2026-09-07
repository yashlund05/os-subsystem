#ifndef NEUROOS_ALLOCATOR_INTERFACE_H
#define NEUROOS_ALLOCATOR_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Allocator type enumeration */
typedef enum {
    ALLOC_ALGO_FIXED_PARTITION = 0,
    ALLOC_ALGO_FIRST_FIT,
    ALLOC_ALGO_BEST_FIT,
    ALLOC_ALGO_BUDDY,
    ALLOC_ALGO_NEUROOS_LITE
} alloc_algo_type_t;

/* Allocation request */
struct alloc_request {
    uint32_t request_id;
    uint32_t task_pid;
    size_t size_bytes;
    uint32_t predicted_lifetime_us; /* Anticipated lifetime tau_k for NeuroOS-Lite affinity binning */
};

/* Allocation handle returned by allocator */
struct alloc_handle {
    uintptr_t base_address;
    size_t allocated_size_bytes;
    size_t requested_size_bytes;
    uint32_t block_id;
    bool success;
};

/* Allocator fragmentation metrics */
struct alloc_metrics {
    size_t total_heap_bytes;
    size_t allocated_bytes;
    size_t free_bytes;
    size_t max_free_block_bytes;
    double external_fragmentation; /* 1 - (max_free_block / sum_free_blocks) */
    double internal_fragmentation; /* sum(allocated - requested) / sum(allocated) */
    double buffer_utilization;    /* active_used_bytes / total_heap_bytes */
    uint64_t total_alloc_requests;
    uint64_t failed_alloc_requests;
    uint64_t total_search_ticks;
};

/* Unified Memory Allocator Operations Interface */
struct allocator_ops {
    const char *name;
    alloc_algo_type_t algo_type;

    int (*init)(void **state, size_t heap_size_bytes, void *config);
    struct alloc_handle (*allocate)(void *state, const struct alloc_request *req);
    int (*deallocate)(void *state, struct alloc_handle handle);
    void (*get_metrics)(void *state, struct alloc_metrics *out_metrics);
    void (*destroy)(void *state);
};

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_ALLOCATOR_INTERFACE_H */
