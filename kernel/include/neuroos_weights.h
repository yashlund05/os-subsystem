#ifndef NEUROOS_WEIGHTS_H
#define NEUROOS_WEIGHTS_H

#include "neuroos_kernel.h"

/* Quantized NeuroOS-Lite Student Policy (Version 1, CRC32 0x554D57AA) */
/* Input Scale s_x: per-feature (16) array, Output Scale s_b2: 0.0000003617 */

static const struct neuroos_quantized_policy neuroos_student_policy = {
    /* w1: (8 x 16) row-major */
    .w1 = {
        /* neuron 0 */ -104, -36, 0, 3, -31, -4, 3, 4, -17, 0, 1, -9, 1, 0, 65, 0,
        /* neuron 1 */ -77, -9, -1, -2, 67, 0, 0, -5, 8, 0, -1, -9, -3, 0, -48, 0,
        /* neuron 2 */ 71, -116, 3, 2, 20, 25, 4, 6, 7, 4, 0, 38, 16, 4, -68, 1,
        /* neuron 3 */ 58, 123, 0, 1, 87, 42, 6, 12, -7, -6, 1, 23, 24, 1, 127, 2,
        /* neuron 4 */ 59, -76, -1, 1, -105, 12, -3, -2, -2, 5, 0, 40, 14, 0, -25, 0,
        /* neuron 5 */ 36, -2, 1, -1, 7, -10, -5, 0, -17, -1, 1, -9, 8, -1, 27, 0,
        /* neuron 6 */ 21, -65, 0, 0, 38, -2, -2, -10, 19, 0, 1, -24, 22, 0, -17, -1,
        /* neuron 7 */ 9, -102, 2, 4, 49, 20, 5, 5, -4, 5, 0, 32, 3, 3, -34, 2
    },
    /* b1: (8) int16 */
    .b1 = {
        -1746, 457, 2254, 690, 5290, -2511, 795, 4332
    },
    /* w2: (1 x 8) int8 */
    .w2 = {
        -82, 37, 109, -127, 69, 34, 31, 103
    },
    /* b2: (1) int32 */
    .b2 = { -843643 },
    .version = 1U,
};

static const float neuroos_scale_s_x[16] = { 0.03937008f, 0.02464214f, 0.00044576f, 0.00094488f, 0.02998994f, 0.00680775f, 0.00189245f, 0.00236220f, 0.00787402f, 0.00039370f, 0.00037678f, 0.00629921f, 0.00787402f, 0.00055490f, 0.02464214f, 0.00039228f };
#define NEUROOS_SCALE_S_W1    8.725438354688941e-05f
#define NEUROOS_SCALE_S_B1    8.725438354688941e-05f
#define NEUROOS_SCALE_S_W2    0.00414529466253566f
#define NEUROOS_SCALE_S_B2    3.6169513039976e-07f

#endif /* NEUROOS_WEIGHTS_H */
