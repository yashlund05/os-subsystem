"""Pure-integer reference forward inference for NeuroOS 16->8->1 micro-core.

Implements integer-only arithmetic (int8, int16, int32) with zero floating-point operations
in the fast path, exactly matching kernel/inference/micro_infer.c.
"""

from __future__ import annotations

from typing import Union

import numpy as np

INPUT_DIM = 16
HIDDEN_DIM = 8
OUTPUT_DIM = 1


def quantize_features_to_int8(features: np.ndarray, s_x: float) -> np.ndarray:
    """Converts continuous float features to int8 using calibrated input scale s_x.

    features: shape (..., 16) float
    s_x: scale factor (e.g. 0.05)
    returns: (..., 16) int8 array in [-128, 127]
    """
    scaled = np.round(features / s_x)
    return np.clip(scaled, -128, 127).astype(np.int8)


# Fixed-point requantization parameters for Acc1 -> hidden int8
# Derived from max(ReLU(Acc1)) = 39,953 across rollout observations:
# scale = 127.0 / 39953.0 = 0.003178732
# Fixed shift S = 20, Multiplier M = round(scale * 2^20) = 3333
# Rounding offset: 2^(S-1) = 2^19 = 524288
REQUANT_SHIFT_S = 20
REQUANT_MULT_M = 3333
REQUANT_ROUNDING_OFFSET = 1 << (REQUANT_SHIFT_S - 1)  # 524288


def requantize_acc1_to_int8(acc1: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Requantizes Layer 1 int32 accumulator to int8 using (M, S) fixed-point scaling.

    Formula:
        h_int8 = clip(((acc1 * M) + 2^(S-1)) >> S, 0, 127)
    with M = 3333, S = 20.
    Bit-identical to neuroos_requantize_acc1() in C.
    """
    if isinstance(acc1, (int, np.integer)):
        if acc1 <= 0:
            return 0
        scaled_int = ((int(acc1) * REQUANT_MULT_M) + REQUANT_ROUNDING_OFFSET) >> REQUANT_SHIFT_S
        return int(min(127, max(0, scaled_int)))
    else:
        acc_clamped = np.maximum(0, acc1.astype(np.int64))
        scaled_arr = ((acc_clamped * REQUANT_MULT_M) + REQUANT_ROUNDING_OFFSET) >> REQUANT_SHIFT_S
        return np.clip(scaled_arr, 0, 127).astype(np.int8)


def int8_forward_single(
    features_int8: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: Union[int, np.ndarray],
) -> int:
    """Evaluates 16->8->1 MLP for a single candidate using pure integer arithmetic.

    features_int8: (16,) int8
    w1: (8, 16) int8
    b1: (8,) int16
    w2: (1, 8) or (8,) int8
    b2: scalar int32 (or (1,) int32)

    Returns:
        int32 output score.
    """
    assert features_int8.shape == (16,), f"Expected shape (16,), got {features_int8.shape}"
    w2_flat = w2.reshape(-1)
    b2_val = int(b2[0]) if isinstance(b2, (np.ndarray, list)) else int(b2)

    # Layer 1: h_j = ReLU(sum_i x_i * w1[j, i] + b1[j])
    # Accumulator: int32 (max 16 * 127 * 127 + 32767 = 290,831, fits safely in int32)
    hidden = np.zeros(8, dtype=np.int32)
    for j in range(8):
        acc = int(b1[j])
        for i in range(16):
            acc += int(features_int8[i]) * int(w1[j, i])
        hidden[j] = acc if acc > 0 else 0

    # Layer 2: out = sum_j h_j * w2[j] + b2
    out = b2_val
    for j in range(8):
        out += int(hidden[j]) * int(w2_flat[j])

    return int(out)


def int8_forward_batch(
    features_int8: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: Union[int, np.ndarray],
) -> np.ndarray:
    """Evaluates 16->8->1 MLP over a batch of candidates using pure integer arithmetic.

    features_int8: (N, 16) int8
    w1: (8, 16) int8
    b1: (8,) int16
    w2: (1, 8) or (8,) int8
    b2: int32 scalar or (1,) array

    Returns:
        (N,) int32 array of candidate scores.
    """
    feat_i32 = features_int8.astype(np.int32)
    w1_i32 = w1.astype(np.int32)
    b1_i32 = b1.astype(np.int32)
    w2_i32 = w2.reshape(1, 8).astype(np.int32)
    b2_val = int(b2[0]) if isinstance(b2, (np.ndarray, list)) else int(b2)

    # Layer 1 matmul: (N, 16) @ (16, 8) -> (N, 8)
    acc1 = feat_i32 @ w1_i32.T + b1_i32
    h = np.maximum(0, acc1)  # int32 ReLU

    # Layer 2 matmul: (N, 8) @ (8, 1) -> (N, 1)
    acc2 = h @ w2_i32.T + b2_val
    return acc2.reshape(-1)
