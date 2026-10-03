"""Piecewise-Linear Model (PLM) and Lookup Tables (LUTs).

Phase 2 Week 4 per docs/TRD.md + docs/Flow.md:
- Quantum scaler LUT: maps priority score -> dynamic quantum in [q_min, q_max].
- Lifetime-bin LUT: maps predicted lifetime tau_k -> affinity band index.
- Pure integer tables for kernel use; Python builder only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class QuantumLUT:
    q_min_us: int = 1000
    q_max_us: int = 50000
    num_bins: int = 32
    table: np.ndarray | None = None

    def build(self) -> np.ndarray:
        # Monotonic mapping: low score (urgent/short) -> small quantum (responsive),
        # high score (batch/steady) -> large quantum (fewer switches).
        # Score domain is int32 quantized output; we normalize bin index linearly.
        table = np.linspace(self.q_min_us, self.q_max_us, self.num_bins).astype(np.int32)
        self.table = table
        return table

    def lookup(self, score_bin: int) -> int:
        if self.table is None:
            self.build()
        assert self.table is not None
        idx = int(np.clip(score_bin, 0, self.num_bins - 1))
        return int(self.table[idx])

    def score_to_quantum(
        self, score: float, score_min: float = -5.0, score_max: float = 5.0
    ) -> int:
        if self.table is None:
            self.build()
        frac = (float(score) - score_min) / max(1e-9, (score_max - score_min))
        frac = float(np.clip(frac, 0.0, 1.0))
        idx = int(frac * (self.num_bins - 1))
        return self.lookup(idx)


def build_quantum_lut(
    q_min_us: int = 1000, q_max_us: int = 50000, num_bins: int = 32
) -> QuantumLUT:
    lut = QuantumLUT(q_min_us=q_min_us, q_max_us=q_max_us, num_bins=num_bins)
    lut.build()
    return lut


def lifetime_to_band(tau_us: int, band_edges_us: List[int] | None = None) -> int:
    """Map predicted lifetime tau_k to affinity band index (Week 6 helper, defined here for reuse)."""
    if band_edges_us is None:
        band_edges_us = [5000, 20000, 100000, 500000]
    for i, edge in enumerate(band_edges_us):
        if tau_us < edge:
            return i
    return len(band_edges_us)
