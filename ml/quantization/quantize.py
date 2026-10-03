"""Symmetric 8-bit quantization for 16->8->1 student MLP.

Phase 2 Week 4 per docs/TRD.md section 3 and docs/Schema.md section 3:
- w1 (16x8) int8 in [-128, 127], b1 (8,) int16, w2 (8x1) int8, b2 (1,) int32.
- 32-bit accumulator, zero floating-point ops in fast path.
- Packs into struct neuroos_quantized_policy layout with version + CRC32 checksum.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

INPUT_DIM = 16
HIDDEN_DIM = 8
OUTPUT_DIM = 1


@dataclass
class QuantizationScales:
    s_w1: float = 0.05
    s_b1: float = 0.01
    s_w2: float = 0.05
    s_b2: float = 0.001
    s_x: float = 0.05  # input feature scale (float feature in [0,5] -> int8)


def _symmetric_scale(arr: np.ndarray, bits: int = 8) -> float:
    max_abs = float(np.max(np.abs(arr))) if arr.size else 1.0
    if max_abs == 0:
        return 1.0
    qmax = float(2 ** (bits - 1) - 1)
    return max_abs / qmax


def quantize_student(
    params: Dict[str, np.ndarray],
    version: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Quantize FP32 student params to integer policy.

    Expects keys: layer_0_weight (8,16), layer_0_bias (8,),
    layer_1_weight (1,8), layer_1_bias (1,).
    Returns (qpolicy dict of int arrays, metadata with scales/checksum).
    """
    w1 = np.asarray(params["layer_0_weight"], dtype=np.float64)  # (8,16)
    b1 = np.asarray(params["layer_0_bias"], dtype=np.float64)
    w2 = np.asarray(params["layer_1_weight"], dtype=np.float64)  # (1,8)
    b2 = np.asarray(params["layer_1_bias"], dtype=np.float64)
    assert w1.shape == (8, 16), f"w1 shape {w1.shape} != (8,16)"
    assert b1.shape == (8,), f"b1 shape {b1.shape}"
    assert w2.shape == (1, 8), f"w2 shape {w2.shape} != (1,8)"

    s_w1 = _symmetric_scale(w1, 8)
    s_w2 = _symmetric_scale(w2, 8)
    # Biases quantized against (s_x * s_w) product per TRD fixed-point chain
    s_x = 0.05
    s_b1 = s_x * s_w1 if s_x * s_w1 > 0 else 0.01
    s_b2 = 0.001

    q_w1 = np.clip(np.round(w1 / s_w1), -128, 127).astype(np.int8)
    q_b1 = np.clip(np.round(b1 / s_b1), -32768, 32767).astype(np.int16)
    q_w2 = np.clip(np.round(w2 / s_w2), -128, 127).astype(np.int8)
    q_b2 = np.round(b2 / s_b2).astype(np.int32)

    # CRC32 over raw bytes (matches kernel checksum field)
    crc = zlib.crc32(q_w1.tobytes())
    crc = zlib.crc32(q_b1.tobytes(), crc)
    crc = zlib.crc32(q_w2.tobytes(), crc)
    crc = zlib.crc32(q_b2.tobytes(), crc)
    crc &= 0xFFFFFFFF

    qpolicy = {"w1": q_w1, "b1": q_b1, "w2": q_w2, "b2": q_b2}
    meta = {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "scales": {"s_w1": s_w1, "s_b1": s_b1, "s_w2": s_w2, "s_b2": s_b2, "s_x": s_x},
        "version": int(version),
        "checksum": int(crc),
    }
    return qpolicy, meta


def quantized_forward_int(
    qpolicy: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    x_float: np.ndarray,
) -> np.ndarray:
    """Integer-faithful forward using int32 accumulators (verification only).

    Mirrors kernel/inference/micro_infer.c fixed-point chain:
      x_q = round(x / s_x); acc1 = sum(x_q*w1)+round(b1/s_x/s_w1)... simplified
    Here we emulate with float rescaling of integer MACs to validate fidelity.
    """
    s = meta["scales"]
    x_q = np.clip(np.round(x_float / s["s_x"]), -128, 127).astype(np.int32)
    w1 = qpolicy["w1"].astype(np.int32)  # (8,16)
    b1 = qpolicy["b1"].astype(np.int32)
    w2 = qpolicy["w2"].astype(np.int32)  # (1,8)
    b2 = qpolicy["b2"].astype(np.int32)
    # Layer1: acc in units of s_x*s_w1
    acc1 = x_q @ w1.T + np.round(b1 / 1.0).astype(np.int32) * 1
    # Note: b1 was quantized with scale s_b1 = s_x*s_w1, and x_q*w1 accumulates
    # in same unit, so direct add is faithful.
    h = np.maximum(0, acc1)  # int ReLU
    # Layer2: h is in units s_x*s_w1; w2 in s_w2; product in s_x*s_w1*s_w2
    acc2 = h @ w2.T  # int32
    # Rescale to output float: out = acc2 * s_x*s_w1*s_w2 + b2*s_b2
    out = acc2.astype(np.float64) * (s["s_x"] * s["s_w1"] * s["s_w2"]) + b2.astype(
        np.float64
    ) * s["s_b2"]
    # Add b1 contribution already inside h; b1 rescale handled via acc1 units:
    # acc1 float = acc1 * s_x*s_w1 ; ReLU preserved scale.
    return out.reshape(-1)


def save_quantized_policy(
    qpolicy: Dict[str, np.ndarray], meta: Dict[str, Any], out_dir: str, prefix: str = "neuroos_student"
) -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{prefix}_int8.npz"
    json_path = out / f"{prefix}_int8.json"
    np.savez(npz_path, w1=qpolicy["w1"], b1=qpolicy["b1"], w2=qpolicy["w2"], b2=qpolicy["b2"])
    serializable = {
        **meta,
        "scales": {k: float(v) for k, v in meta["scales"].items()},
        "version": int(meta["version"]),
        "checksum": int(meta["checksum"]),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return npz_path, json_path


def c_header_bytes(qpolicy: Dict[str, np.ndarray], meta: Dict[str, Any]) -> bytes:
    """Pack policy as C struct bytes (w1[128]i8, b1[8]i16, w2[8]i8, b2[1]i32, version u32)."""
    data = struct.pack(
        "<128b8h8b1iI",
        *qpolicy["w1"].reshape(-1).tolist(),
        *qpolicy["b1"].reshape(-1).tolist(),
        *qpolicy["w2"].reshape(-1).tolist(),
        *qpolicy["b2"].reshape(-1).tolist(),
        int(meta["version"]),
    )
    return data
