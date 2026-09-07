#include <stdio.h>
#include <assert.h>
#include "telemetry_event.h"
#include "neuroos_kernel.h"
#include "scheduler_interface.h"
#include "allocator_interface.h"
#include "ring_buffer.h"

int main(void) {
    printf("[TEST] Running NeuroOS-Lite C Header Scaffolding Validation...\n");

    /* 1. Verify telemetry_event struct packing and size */
    assert(sizeof(struct task_telemetry) == 16);
    printf("  [PASS] struct task_telemetry is exactly 16 bytes (size = %zu)\n", sizeof(struct task_telemetry));

    /* 2. Verify quantized policy dimensions */
    assert(NEUROOS_INPUT_DIM == 16);
    assert(NEUROOS_HIDDEN_DIM == 8);
    assert(NEUROOS_OUTPUT_DIM == 1);
    printf("  [PASS] Micro-inference MLP topology: 16 -> 8 -> 1\n");

    /* 3. Verify guardrail constants */
    assert(NEUROOS_MAX_QUEUE_DEPTH == 1024);
    assert(NEUROOS_DRIFT_THRESHOLD == 3000);
    printf("  [PASS] Guardrail thresholds: Max Q = 1024, Drift = 3.0 sigma\n");

    /* 4. Verify scheduler and allocator enums */
    assert(SCHED_ALGO_NEUROOS_LITE == 5);
    assert(ALLOC_ALGO_NEUROOS_LITE == 4);
    printf("  [PASS] Scheduler and Allocator algorithm definitions aligned\n");

    printf("[SUCCESS] All NeuroOS-Lite C interface headers validated successfully.\n");
    return 0;
}
