"""Performance & latency regression tests – Phase 2, Week 4.

Validates:
- Python-layer micro-inference latency (surrogate for the C rdtsc harness).
- Quantized forward-pass reproducibility across multiple calls.
- Overhead stays below the soft guard threshold of 50 µs in the Python emulation layer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 42
_N_WARMUP = 50
_N_ITERS = 500

# Soft wall for the Python numpy emulation (NOT the C rdtsc target).
# The C rdtsc target on an isolated P-core is ≤45 ns.
# Python overhead is expected to be ~200–2000 ns; we assert < 100 µs (100_000 ns)
# to catch catastrophic regressions only.
_PYTHON_SOFT_WALL_NS = 100_000  # 100 µs


def _make_random_int8_policy(rng: np.random.Generator):
    w1 = (rng.integers(-10, 10, size=(8, 16))).astype(np.int8)
    b1 = np.zeros(8, dtype=np.int16)
    w2 = (rng.integers(-10, 10, size=(1, 8))).astype(np.int8)
    b2 = np.zeros(1, dtype=np.int32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _make_random_fp_params(rng: np.random.Generator):
    return {
        "layer_0_weight": (rng.standard_normal((8, 16)) * 0.3).astype(np.float64),
        "layer_0_bias": np.zeros(8),
        "layer_1_weight": (rng.standard_normal((1, 8)) * 0.3).astype(np.float64),
        "layer_1_bias": np.zeros(1),
    }


# ---------------------------------------------------------------------------
# 1. Quantized forward latency regression
# ---------------------------------------------------------------------------


def test_quantized_forward_latency_below_soft_wall() -> None:
    """Mean Python-layer quantized-forward latency must be < 100 µs."""
    from ml.quantization.quantize import quantize_student, quantized_forward_int  # type: ignore

    rng = np.random.default_rng(_SEED)
    params = _make_random_fp_params(rng)
    feat = (rng.random((1, 16)) * 5.0).astype(np.float64)
    qpolicy, meta = quantize_student(params, version=1)

    # Warm-up
    for _ in range(_N_WARMUP):
        quantized_forward_int(qpolicy, meta, feat)

    latencies_ns: list[int] = []
    for _ in range(_N_ITERS):
        t0 = time.perf_counter_ns()
        quantized_forward_int(qpolicy, meta, feat)
        latencies_ns.append(time.perf_counter_ns() - t0)

    mean_ns = float(np.mean(latencies_ns))
    p99_ns = float(np.percentile(latencies_ns, 99))

    assert mean_ns < _PYTHON_SOFT_WALL_NS, (
        f"Mean Python-layer latency {mean_ns:.0f} ns exceeds soft wall "
        f"{_PYTHON_SOFT_WALL_NS} ns (regression detected)"
    )
    # P99 allowed up to 10× the soft wall (spikes due to GC etc.)
    assert p99_ns < _PYTHON_SOFT_WALL_NS * 10, (
        f"P99 latency {p99_ns:.0f} ns > 10× soft wall (severe regression)"
    )


# ---------------------------------------------------------------------------
# 2. Student FP forward latency regression
# ---------------------------------------------------------------------------


def test_fp_student_forward_latency() -> None:
    """FP32/64 student forward must complete < 500 µs mean on single feature vector."""
    from ml.distillation.distiller import numpy_student_forward  # type: ignore

    rng = np.random.default_rng(_SEED + 1)
    params = _make_random_fp_params(rng)
    feat = (rng.random((1, 16)) * 5.0).astype(np.float64)

    for _ in range(_N_WARMUP):
        numpy_student_forward(params, feat)

    lats: list[int] = []
    for _ in range(_N_ITERS):
        t0 = time.perf_counter_ns()
        numpy_student_forward(params, feat)
        lats.append(time.perf_counter_ns() - t0)

    mean_ns = float(np.mean(lats))
    assert mean_ns < 500_000, (
        f"FP student forward mean {mean_ns:.0f} ns exceeds 500 µs soft wall"
    )


# ---------------------------------------------------------------------------
# 3. Reproducibility – deterministic outputs
# ---------------------------------------------------------------------------


def test_quantized_forward_deterministic() -> None:
    """Same inputs must produce identical outputs across 100 calls (no randomness)."""
    from ml.quantization.quantize import quantize_student, quantized_forward_int  # type: ignore

    rng = np.random.default_rng(_SEED)
    params = _make_random_fp_params(rng)
    feat = (rng.random((4, 16)) * 5.0).astype(np.float64)
    qpolicy, meta = quantize_student(params, version=1)

    first = quantized_forward_int(qpolicy, meta, feat).copy()
    for _ in range(99):
        result = quantized_forward_int(qpolicy, meta, feat)
        np.testing.assert_array_equal(
            result, first, err_msg="Quantized forward is non-deterministic"
        )


def test_fp_student_forward_deterministic() -> None:
    """FP student forward must be deterministic for identical inputs."""
    from ml.distillation.distiller import numpy_student_forward  # type: ignore

    rng = np.random.default_rng(_SEED)
    params = _make_random_fp_params(rng)
    feat = (rng.random((4, 16)) * 5.0).astype(np.float64)

    first = numpy_student_forward(params, feat).copy()
    for _ in range(50):
        result = numpy_student_forward(params, feat)
        np.testing.assert_allclose(
            result, first, rtol=0, atol=0,
            err_msg="FP student forward is non-deterministic"
        )


# ---------------------------------------------------------------------------
# 4. Batch throughput – sanity check
# ---------------------------------------------------------------------------


def test_quantized_forward_batch_throughput() -> None:
    """Batched inference of 1024 vectors must finish in < 1 second total."""
    from ml.quantization.quantize import quantize_student, quantized_forward_int  # type: ignore

    rng = np.random.default_rng(_SEED + 2)
    params = _make_random_fp_params(rng)
    feat = (rng.random((1024, 16)) * 5.0).astype(np.float64)
    qpolicy, meta = quantize_student(params, version=1)

    t0 = time.perf_counter()
    quantized_forward_int(qpolicy, meta, feat)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, (
        f"Batch of 1024 vectors took {elapsed:.3f}s > 1 s threshold (throughput regression)"
    )
