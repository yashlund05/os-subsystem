/*
 * Overhead micro-benchmark harness (Phase 4, Week 8).
 * Measures neuroos_micro_infer_score cycles via rdtsc_ordered on isolated core.
 * Build: compiled via CMake target overhead_bench. Run pinned: taskset -c 2 ./overhead_bench
 * Target: < 150 cycles (~45-50 ns @ 3.0 GHz). Prints CSV to stdout (REAL measurements).
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "neuroos_kernel.h"
#include "micro_infer.h"

#define ITERS 100000

int main(void) {
    struct neuroos_quantized_policy policy;
    int8_t features[NEUROOS_INPUT_DIM];
    int i, j;
    uint64_t t0, t1;
    uint64_t total = 0;
    uint64_t min_c = (uint64_t)-1;
    uint64_t max_c = 0;
    int32_t sink = 0;

    for (i = 0; i < NEUROOS_INPUT_DIM * NEUROOS_HIDDEN_DIM; i++) policy.w1[i] = (int8_t)((i % 5) - 2);
    for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) policy.b1[j] = (int16_t)(j * 10);
    for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) policy.w2[j] = (int8_t)((j % 3) - 1);
    policy.b2[0] = 100;
    policy.version = 1;
    for (i = 0; i < NEUROOS_INPUT_DIM; i++) features[i] = (int8_t)(i - 8);

    /* Warmup */
    for (i = 0; i < 1000; i++) sink += neuroos_micro_infer_score(features, &policy);

    for (i = 0; i < ITERS; i++) {
        t0 = neuroos_rdtsc_ordered();
        sink += neuroos_micro_infer_score(features, &policy);
        t1 = neuroos_rdtsc_ordered();
        {
            uint64_t d = t1 - t0;
            total += d;
            if (d < min_c) min_c = d;
            if (d > max_c) max_c = d;
        }
    }

    printf("iters,mean_cycles,min_cycles,max_cycles,sink\n");
    printf("%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n", ITERS, total / (uint64_t)ITERS,
           min_c, max_c, (int)sink);
    return 0;
}
