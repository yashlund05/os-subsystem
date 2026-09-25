#include "include/ring_buffer.h"
#include <stdlib.h>
#include <string.h>

/* Helper to check if a number is a power of two */
static bool is_power_of_two(uint32_t n) {
    return (n != 0) && ((n & (n - 1)) == 0);
}

/* Helper to find next power of two */
static uint32_t next_power_of_two(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

struct spsc_ring_buffer *spsc_ring_buffer_create(uint32_t capacity) {
    if (capacity == 0) {
        capacity = NEUROOS_RING_BUFFER_DEFAULT_CAPACITY;
    }
    
    if (!is_power_of_two(capacity)) {
        capacity = next_power_of_two(capacity);
    }
    
    struct spsc_ring_buffer *rb = (struct spsc_ring_buffer *)malloc(
        sizeof(struct spsc_ring_buffer) + capacity * sizeof(struct task_telemetry)
    );
    
    if (!rb) {
        return NULL;
    }
    
    rb->capacity = capacity;
    rb->mask = capacity - 1;
    rb->head = 0;
    rb->tail = 0;
    
    return rb;
}

void spsc_ring_buffer_destroy(struct spsc_ring_buffer *rb) {
    if (rb) {
        free(rb);
    }
}

bool spsc_ring_buffer_push(struct spsc_ring_buffer *rb, const struct task_telemetry *item) {
    if (!rb || !item) return false;
    
    uint32_t head = __atomic_load_n(&rb->head, __ATOMIC_RELAXED);
    uint32_t tail = __atomic_load_n(&rb->tail, __ATOMIC_ACQUIRE);
    
    if ((head - tail) >= rb->capacity) {
        return false; /* Full */
    }
    
    rb->entries[head & rb->mask] = *item;
    
    __atomic_store_n(&rb->head, head + 1, __ATOMIC_RELEASE);
    
    return true;
}

bool spsc_ring_buffer_pop(struct spsc_ring_buffer *rb, struct task_telemetry *item) {
    if (!rb || !item) return false;
    
    uint32_t tail = __atomic_load_n(&rb->tail, __ATOMIC_RELAXED);
    uint32_t head = __atomic_load_n(&rb->head, __ATOMIC_ACQUIRE);
    
    if (head == tail) {
        return false; /* Empty */
    }
    
    *item = rb->entries[tail & rb->mask];
    
    __atomic_store_n(&rb->tail, tail + 1, __ATOMIC_RELEASE);
    
    return true;
}

uint32_t spsc_ring_buffer_count(const struct spsc_ring_buffer *rb) {
    if (!rb) return 0;
    uint32_t head = __atomic_load_n(&rb->head, __ATOMIC_ACQUIRE);
    uint32_t tail = __atomic_load_n(&rb->tail, __ATOMIC_ACQUIRE);
    return head - tail;
}

bool spsc_ring_buffer_is_full(const struct spsc_ring_buffer *rb) {
    if (!rb) return false;
    return spsc_ring_buffer_count(rb) >= rb->capacity;
}

bool spsc_ring_buffer_is_empty(const struct spsc_ring_buffer *rb) {
    if (!rb) return true;
    return spsc_ring_buffer_count(rb) == 0;
}
