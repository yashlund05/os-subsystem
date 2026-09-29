"""Daemon LUT builder entry point (wraps ml.quantization.lut)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ml.quantization.lut import build_quantum_lut


def build_kernel_tables(
    q_min_us: int = 1000, q_max_us: int = 50000, num_bins: int = 32
) -> Dict[str, Any]:
    lut = build_quantum_lut(q_min_us=q_min_us, q_max_us=q_max_us, num_bins=num_bins)
    table = lut.table if lut.table is not None else np.array([], dtype=np.int32)
    return {
        "q_min_us": q_min_us,
        "q_max_us": q_max_us,
        "num_bins": num_bins,
        "table": [int(v) for v in table.tolist()],
    }
