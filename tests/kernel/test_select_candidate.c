/*
 * test_select_candidate.c — Standalone C test harness for neuroos_select_candidate()
 *
 * Remediation item 0.3: proves the dispatch logic in neuroos_sched.c is exercised
 * and correct without requiring a real Linux kernel or sched_ext BPF program.
 *
 * Build (GCC/Clang):
 *   gcc -std=c11 -I../../kernel/include -I../../kernel/sched_ext \
 *       test_select_candidate.c \
 *       ../../kernel/sched_ext/neuroos_sched.c \
 *       ../../kernel/inference/micro_infer.c \
 *       -o test_select_candidate && ./test_select_candidate
 *
 * Expected output:  all PASS lines, exit code 0.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

/* Pull in the sched header (which includes neuroos_kernel.h) */
#include "../../kernel/sched_ext/neuroos_sched.h"

/* -------------------------------------------------------------------------
 * Minimal guardrail implementation (normally compiled from kernel/sched_ext).
 * Returns true (pass) unless queue depth exceeds threshold or drift > 3 sigma.
 * ------------------------------------------------------------------------- */
bool neuroos_guardrail_check(uint32_t queue_depth, int32_t running_drift_milli_sigma) {
    if (queue_depth > NEUROOS_MAX_QUEUE_DEPTH) return false;
    if (running_drift_milli_sigma > 3000 || running_drift_milli_sigma < -3000) return false;
    return true;
}

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */
static void make_policy(struct neuroos_quantized_policy *p) {
    int i, j;
    for (i = 0; i < NEUROOS_INPUT_DIM * NEUROOS_HIDDEN_DIM; i++)
        p->w1[i] = (int8_t)((i % 5) - 2);
    for (j = 0; j < NEUROOS_HIDDEN_DIM; j++)
        p->b1[j] = (int16_t)(j * 10);
    for (j = 0; j < NEUROOS_HIDDEN_DIM; j++)
        p->w2[j] = (int8_t)((j % 3) - 1);
    p->b2[0] = 100;
    p->version = 1;
}

static void make_entry(struct neuroos_ready_entry *e, uint32_t pid, int8_t fill) {
    int k;
    e->pid = pid;
    for (k = 0; k < NEUROOS_INPUT_DIM; k++)
        e->features[k] = fill;
}

#define ASSERT_EQ(name, got, expected) \
    do { \
        if ((got) == (expected)) { \
            printf("PASS  %s: got %d\n", (name), (int)(got)); \
        } else { \
            printf("FAIL  %s: expected %d, got %d\n", (name), (int)(expected), (int)(got)); \
            failures++; \
        } \
    } while (0)

#define ASSERT_GE(name, got, lo) \
    do { \
        if ((got) >= (lo)) { \
            printf("PASS  %s: got %u >= %u\n", (name), (unsigned)(got), (unsigned)(lo)); \
        } else { \
            printf("FAIL  %s: expected >= %u, got %u\n", (name), (unsigned)(lo), (unsigned)(got)); \
            failures++; \
        } \
    } while (0)

#define ASSERT_LE(name, got, hi) \
    do { \
        if ((got) <= (hi)) { \
            printf("PASS  %s: got %u <= %u\n", (name), (unsigned)(got), (unsigned)(hi)); \
        } else { \
            printf("FAIL  %s: expected <= %u, got %u\n", (name), (unsigned)(hi), (unsigned)(got)); \
            failures++; \
        } \
    } while (0)

#define ASSERT_TRUE(name, cond) \
    do { \
        if (cond) { \
            printf("PASS  %s\n", (name)); \
        } else { \
            printf("FAIL  %s: condition was false\n", (name)); \
            failures++; \
        } \
    } while (0)

/* -------------------------------------------------------------------------
 * Test cases
 * ------------------------------------------------------------------------- */

/* TC-1: Normal dispatch — 4 candidates, each with distinct feature fills.
 * The function must return a valid index in [0, 3] and populate out_decision. */
static int tc1_normal_dispatch(void) {
    int failures = 0;
    struct neuroos_quantized_policy policy;
    struct neuroos_ready_entry entries[4];
    struct neuroos_decision dec;
    int ret;

    make_policy(&policy);
    make_entry(&entries[0], 101, -8);
    make_entry(&entries[1], 102,  0);
    make_entry(&entries[2], 103,  8);
    make_entry(&entries[3], 104, 64);

    memset(&dec, 0, sizeof(dec));
    ret = neuroos_select_candidate(entries, 4, &policy,
                                   /*queue_depth=*/10,
                                   /*drift=*/0,
                                   &dec, 1000, 50000);

    printf("\n--- TC-1: Normal dispatch (4 candidates) ---\n");
    ASSERT_TRUE("ret in [0,3]", ret >= 0 && ret <= 3);
    ASSERT_TRUE("fallback not engaged", !dec.fallback_engaged);
    ASSERT_TRUE("selected_pid non-zero", dec.selected_pid != 0);
    ASSERT_GE("quantum_us >= q_min", dec.quantum_us, 1000u);
    ASSERT_LE("quantum_us <= q_max", dec.quantum_us, 50000u);
    ASSERT_EQ("reason_code is 0", dec.reason_code, 0);
    return failures;
}

/* TC-2: Guardrail trips on queue depth > NEUROOS_MAX_QUEUE_DEPTH (1024).
 * Must return -1 and set fallback_engaged=true with reason_code=1. */
static int tc2_guardrail_queue(void) {
    int failures = 0;
    struct neuroos_quantized_policy policy;
    struct neuroos_ready_entry entries[2];
    struct neuroos_decision dec;
    int ret;

    make_policy(&policy);
    make_entry(&entries[0], 201, 10);
    make_entry(&entries[1], 202, 20);
    memset(&dec, 0, sizeof(dec));

    ret = neuroos_select_candidate(entries, 2, &policy,
                                   /*queue_depth=*/2048,  /* exceeds 1024 */
                                   /*drift=*/0,
                                   &dec, 1000, 50000);

    printf("\n--- TC-2: Guardrail trips on excessive queue depth ---\n");
    ASSERT_EQ("ret is -1", ret, -1);
    ASSERT_TRUE("fallback_engaged", dec.fallback_engaged);
    ASSERT_EQ("reason_code is 1", dec.reason_code, 1);
    ASSERT_EQ("quantum_us is q_min", dec.quantum_us, 1000u);
    return failures;
}

/* TC-3: Guardrail trips on excessive drift (> 3000 milli-sigma).
 * Must return -1 with reason_code=2 (drift condition). */
static int tc3_guardrail_drift(void) {
    int failures = 0;
    struct neuroos_quantized_policy policy;
    struct neuroos_ready_entry entries[2];
    struct neuroos_decision dec;
    int ret;

    make_policy(&policy);
    make_entry(&entries[0], 301, 5);
    make_entry(&entries[1], 302, 5);
    memset(&dec, 0, sizeof(dec));

    ret = neuroos_select_candidate(entries, 2, &policy,
                                   /*queue_depth=*/5,
                                   /*drift=*/5000,  /* > 3000 milli-sigma */
                                   &dec, 1000, 50000);

    printf("\n--- TC-3: Guardrail trips on excessive drift ---\n");
    ASSERT_EQ("ret is -1", ret, -1);
    ASSERT_TRUE("fallback_engaged", dec.fallback_engaged);
    ASSERT_EQ("reason_code is 2", dec.reason_code, 2);
    return failures;
}

/* TC-4: Single candidate — must always be selected (index 0). */
static int tc4_single_candidate(void) {
    int failures = 0;
    struct neuroos_quantized_policy policy;
    struct neuroos_ready_entry entries[1];
    struct neuroos_decision dec;
    int ret;

    make_policy(&policy);
    make_entry(&entries[0], 401, 0);
    memset(&dec, 0, sizeof(dec));

    ret = neuroos_select_candidate(entries, 1, &policy,
                                   0, 0, &dec, 1000, 50000);

    printf("\n--- TC-4: Single candidate ---\n");
    ASSERT_EQ("ret is 0", ret, 0);
    ASSERT_EQ("selected_pid", (int)dec.selected_pid, 401);
    ASSERT_TRUE("fallback not engaged", !dec.fallback_engaged);
    return failures;
}

/* TC-5: Zero candidates — must return -1 immediately. */
static int tc5_zero_candidates(void) {
    int failures = 0;
    struct neuroos_ready_entry entries[1];   /* unused but avoids NULL */
    struct neuroos_decision dec;
    struct neuroos_quantized_policy policy;
    int ret;

    make_policy(&policy);
    make_entry(&entries[0], 501, 0);
    memset(&dec, 0, sizeof(dec));

    ret = neuroos_select_candidate(entries, 0, &policy,
                                   0, 0, &dec, 1000, 50000);

    printf("\n--- TC-5: Zero candidates ---\n");
    ASSERT_EQ("ret is -1", ret, -1);
    return failures;
}

/* TC-6: NULL pointer arguments — must return -1 defensively. */
static int tc6_null_args(void) {
    int failures = 0;
    struct neuroos_quantized_policy policy;
    struct neuroos_ready_entry entries[2];
    struct neuroos_decision dec;
    int ret;

    make_policy(&policy);
    make_entry(&entries[0], 601, 0);

    printf("\n--- TC-6: NULL argument rejection ---\n");

    ret = neuroos_select_candidate(NULL, 2, &policy, 0, 0, &dec, 1000, 50000);
    ASSERT_EQ("NULL entries → -1", ret, -1);

    ret = neuroos_select_candidate(entries, 2, NULL, 0, 0, &dec, 1000, 50000);
    ASSERT_EQ("NULL policy → -1", ret, -1);

    ret = neuroos_select_candidate(entries, 2, &policy, 0, 0, NULL, 1000, 50000);
    ASSERT_EQ("NULL out_decision → -1", ret, -1);

    return failures;
}

/* TC-7: Determinism — same inputs, same output across 100 calls. */
static int tc7_determinism(void) {
    int failures = 0;
    struct neuroos_quantized_policy policy;
    struct neuroos_ready_entry entries[4];
    struct neuroos_decision dec0, dec;
    int ret0, ret, call;

    make_policy(&policy);
    make_entry(&entries[0], 701, -4);
    make_entry(&entries[1], 702,  0);
    make_entry(&entries[2], 703,  4);
    make_entry(&entries[3], 704, 12);

    memset(&dec0, 0, sizeof(dec0));
    ret0 = neuroos_select_candidate(entries, 4, &policy, 10, 0, &dec0, 1000, 50000);

    printf("\n--- TC-7: Determinism (100 identical calls) ---\n");
    for (call = 1; call < 100; call++) {
        memset(&dec, 0, sizeof(dec));
        ret = neuroos_select_candidate(entries, 4, &policy, 10, 0, &dec, 1000, 50000);
        if (ret != ret0 || dec.selected_pid != dec0.selected_pid ||
            dec.quantum_us != dec0.quantum_us) {
            printf("FAIL  call %d differs from call 0\n", call);
            failures++;
            break;
        }
    }
    if (!failures)
        printf("PASS  all 100 calls produced identical results (ret=%d, pid=%u, q=%u)\n",
               ret0, dec0.selected_pid, dec0.quantum_us);
    return failures;
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void) {
    int total_failures = 0;

    printf("=== neuroos_select_candidate test harness (remediation item 0.3) ===\n");

    total_failures += tc1_normal_dispatch();
    total_failures += tc2_guardrail_queue();
    total_failures += tc3_guardrail_drift();
    total_failures += tc4_single_candidate();
    total_failures += tc5_zero_candidates();
    total_failures += tc6_null_args();
    total_failures += tc7_determinism();

    printf("\n=== Result: %d failure(s) ===\n", total_failures);
    return total_failures == 0 ? 0 : 1;
}
