#include "include/ring_buffer.h"
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
/* MSVC does not provide GCC __atomic_* builtins. This is an SPSC queue, so
 * volatile head/tail plus compiler barriers are sufficient on x86/x64 TSO.
 * Documented per docs/Rules.md: portable fallback, no locks. */
#include <intrin.h>
static inline uint32_t rb_load_relaxed(const volatile uint32_t *p) {
    return *p;
}
static inline uint32_t rb_load_acquire(const volatile uint32_t *p) {
    uint32_t v = *p;
    _ReadBarrier();
    return v;
}
static inline void rb_store_release(volatile uint32_t *p, uint32_t v) {
    _WriteBarrier();
    *p = v;
}
#else
static inline uint32_t rb_load_relaxed(const volatile uint32_t *p) {
    return __atomic_load_n(p, __ATOMIC_RELAXED);
}
static inline uint32_t rb_load_acquire(const volatile uint32_t *p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static inline void rb_store_release(volatile uint32_t *p, uint32_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
#endif

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

    uint32_t head = rb_load_relaxed(&rb->head);
    uint32_t tail = rb_load_acquire(&rb->tail);

    if ((head - tail) >= rb->capacity) {
        return false; /* Full */
    }

    rb->entries[head & rb->mask] = *item;

    rb_store_release(&rb->head, head + 1);

    return true;
}

bool spsc_ring_buffer_pop(struct spsc_ring_buffer *rb, struct task_telemetry *item) {
    if (!rb || !item) return false;

    uint32_t tail = rb_load_relaxed(&rb->tail);
    uint32_t head = rb_load_acquire(&rb->head);

    if (head == tail) {
        return false; /* Empty */
    }

    *item = rb->entries[tail & rb->mask];

    rb_store_release(&rb->tail, tail + 1);

    return true;
}

uint32_t spsc_ring_buffer_count(const struct spsc_ring_buffer *rb) {
    if (!rb) return 0;
    uint32_t head = rb_load_acquire(&rb->head);
    uint32_t tail = rb_load_acquire(&rb->tail);
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
