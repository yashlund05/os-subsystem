#ifndef NEUROOS_TELEMETRY_EVENT_H
#define NEUROOS_TELEMETRY_EVENT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compact 16-Byte Telemetry Struct per Context Switch.
 * Streams hardware performance counters (PMU) and execution progression
 * across the lock-free SPSC ring buffer to the user-space GPU daemon.
 */
#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif

struct
#if defined(__GNUC__) || defined(__clang__)
__attribute__((packed))
#endif
task_telemetry {
    uint32_t pid;                  /* Task identifier (32 bits) */
    uint16_t elapsed_us;            /* Elapsed execution time in microsecond ticks (16 bits) */
    uint16_t cache_misses_delta;   /* L1/LLC cache miss delta since last switch (16 bits) */
    uint16_t branch_mispred_delta; /* Branch misprediction delta (16 bits) */
    uint16_t mem_footprint_kb;     /* Task memory working set in KB (16 bits) */
    uint16_t flags;                /* State flags: burst phase, priority, guardrail status (16 bits) */
};

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

/* Telemetry flag bit definitions */
#define TELEMETRY_FLAG_NONE           0x0000
#define TELEMETRY_FLAG_BURST_START    0x0001
#define TELEMETRY_FLAG_BURST_END      0x0002
#define TELEMETRY_FLAG_IO_BLOCKED     0x0004
#define TELEMETRY_FLAG_PREEMPTED      0x0008
#define TELEMETRY_FLAG_GUARDRAIL_TRIP 0x0010
#define TELEMETRY_FLAG_FALLBACK_ACTIVE 0x0020

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(struct task_telemetry) == 16, "task_telemetry struct must be exactly 16 bytes");
#endif

#ifdef __cplusplus
}
#endif

#endif /* NEUROOS_TELEMETRY_EVENT_H */
