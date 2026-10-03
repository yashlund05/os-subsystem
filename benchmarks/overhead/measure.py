"""Overhead + power measurement (Phase 4 Week 8).

Per docs/Experimental-Protocol.md section 2 + docs/Metrics.md section 3:
- Inference latency via time.perf_counter_ns around NumPy quantized forward
  (C rdtsc harness in kernel/inference/overhead_bench.c for hardware runs).
- Power via Intel RAPL (sysfs) + NVIDIA NVML (pynvml) with graceful unavailable.
- Never fabricates: returns measured values or explicit unavailable flags.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


def measure_quantized_latency_ns(
    qpolicy: Dict[str, np.ndarray],
    scales: Dict[str, float],
    num_iters: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Measure mean/p50/p99 per-inference latency of integer-faithful forward."""
    from ml.quantization.quantize import quantized_forward_int

    rng = np.random.default_rng(seed)
    feats = (rng.random((32, 16)).astype(np.float64) * 5.0).astype(np.float64)
    meta = {"scales": scales}
    # Warmup
    for _ in range(100):
        quantized_forward_int(qpolicy, meta, feats[:1])
    lat: list[int] = []
    for _ in range(num_iters):
        t0 = time.perf_counter_ns()
        quantized_forward_int(qpolicy, meta, feats[:1])
        lat.append(time.perf_counter_ns() - t0)
    arr = np.asarray(lat, dtype=np.float64)
    return {
        "mean_ns": float(np.mean(arr)),
        "p50_ns": float(np.percentile(arr, 50)),
        "p99_ns": float(np.percentile(arr, 99)),
        "iters": int(num_iters),
        "note": "Python emulation; C rdtsc target <=45ns on isolated P-core",
    }


def read_rapl_joules() -> Dict[str, Any]:
    """Read Intel RAPL package energy if sysfs available, else unavailable."""
    base = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    if base.exists():
        try:
            uj = int(base.read_text().strip())
            return {"available": True, "package_uj": uj}
        except Exception as e:
            return {"available": False, "reason": str(e)}
    return {"available": False, "reason": "no RAPL sysfs (non-Linux or VM)"}


def read_nvml() -> Dict[str, Any]:
    """Read NVIDIA NVML power if available, else unavailable."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mw = int(pynvml.nvmlDeviceGetPowerUsage(h))  # milliwatts
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "available": True,
            "power_mw": mw,
            "mem_used_bytes": int(mem.used),
            "mem_total_bytes": int(mem.total),
        }
    except Exception as e:
        return {"available": False, "reason": f"NVML unavailable: {e}"}

