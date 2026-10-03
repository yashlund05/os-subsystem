#ifndef NEUROOS_RING_BUFFER_H
#define NEUROOS_RING_BUFFER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "telemetry_event.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer
 *
 * Dedicated off-path telemetry pipeline:
 * - Producer: Kernel context-switch hook (sched_ext) writing 16-byte task_telemetry structs.
 * - Consumer: User-space GPU training daemon reading batches for DRL optimization.
 * - Capacity: Must be power-of-two for branchless modulo bitmasking.
 */

#define NEUROOS_RING_BUFFER_DEFAULT_CAPACITY 65536

struct spsc_ring_buffer {
    uint32_t capacity;
    uint32_t mask;
    /* Cache-line padded head and tail to prevent false sharing */
    volatile uint32_t head;
    uint8_t pad1[60];
    volatile uint32_t tail;
    uint8_t pad2[60];
    struct task_telemetry entries[];
};

/* Ring buffer operations */
struct spsc_ring_buffer *spsc_ring_buffer_create(uint32_t capacity);
void spsc_ring_buffer_destroy(struct spsc_ring_buffer *rb);

/*
 * Producer: Enqueue telemetry item. Returns true if queued, false if full (dropped).
 * Non-blocking, wait-free.
 */
bool spsc_ring_buffer_push(struct spsc_ring_buffer *rb, const struct task_telemetry *item);

/*
 * Consumer: Dequeue telemetry item. Returns true if read, false if empty.
 */
bool spsc_ring_buffer_pop(struct spsc_ring_buffer *rb, struct task_telemetry *item);

/* Query available items and capacity */
uint32_t spsc_ring_buffer_count(const struct spsc_ring_buffer *rb);
bool spsc_ring_buffer_is_full(const struct spsc_ring_buffer *rb);
bool spsc_ring_buffer_is_empty(const struct spsc_ring_buffer *rb);

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_RING_BUFFER_H */
