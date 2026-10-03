/*
 * NeuroOS-Lite integer-only reference forward implementation (ANSI C).
 * Corresponds to kernel/include/neuroos_kernel.h and quantization/int8_forward.py.
 * Zero floating-point operations in fast path.
 */

#include <stdint.h>
#include <stdbool.h>

#define NEUROOS_INPUT_DIM       16
#define NEUROOS_HIDDEN_DIM      8
#define NEUROOS_OUTPUT_DIM      1

#define NEUROOS_REQUANT_SHIFT_S    20
#define NEUROOS_REQUANT_MULT_M     3333
#define NEUROOS_REQUANT_ROUNDING   (1 << 19) /* 524288 */

/* Requantizes Layer 1 int32 accumulator to int8 using (M, S) fixed-point scaling */
static inline int8_t neuroos_requantize_acc1(int32_t acc1) {
    if (acc1 <= 0) return 0;
    int64_t prod = (int64_t)acc1 * NEUROOS_REQUANT_MULT_M + NEUROOS_REQUANT_ROUNDING;
    int32_t scaled = (int32_t)(prod >> NEUROOS_REQUANT_SHIFT_S);
    if (scaled > 127) scaled = 127;
    return (int8_t)scaled;
}

struct neuroos_quantized_policy {
    int8_t  w1[NEUROOS_INPUT_DIM * NEUROOS_HIDDEN_DIM]; /* Layer 1 weights (8x16 row-major) */
    int16_t b1[NEUROOS_HIDDEN_DIM];                    /* Layer 1 bias (8) */
    int8_t  w2[NEUROOS_HIDDEN_DIM * NEUROOS_OUTPUT_DIM]; /* Layer 2 weights (1x8) */
    int32_t b2[NEUROOS_OUTPUT_DIM];                    /* Layer 2 bias (1) */
    uint32_t version;                                  /* Monotonic version */
};

int32_t neuroos_micro_infer_score(const int8_t features[NEUROOS_INPUT_DIM],
                                  const struct neuroos_quantized_policy *policy) {
    int32_t hidden[NEUROOS_HIDDEN_DIM];
    int i, j;

    /* Layer 1: h_j = ReLU(sum_i x_i * w1[j * 16 + i] + b1[j]) */
    for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) {
        int32_t acc = (int32_t)policy->b1[j];
        for (i = 0; i < NEUROOS_INPUT_DIM; i++) {
            acc += (int32_t)features[i] * (int32_t)policy->w1[j * NEUROOS_INPUT_DIM + i];
        }
        /* Branchless integer ReLU */
        hidden[j] = acc > 0 ? acc : 0;
    }

    /* Layer 2: out = sum_j h_j * w2[j] + b2[0] */
    {
        int32_t out = policy->b2[0];
        for (j = 0; j < NEUROOS_HIDDEN_DIM; j++) {
            out += hidden[j] * (int32_t)policy->w2[j];
        }
        return out;
    }
}
