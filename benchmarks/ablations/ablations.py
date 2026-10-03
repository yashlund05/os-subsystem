"""Ablation studies (Phase 4 Week 8).

Per docs/Experimental-Protocol.md section 5:
- Ablation 1 (PMU): full 16-D vs burst-history-only (zero PMU indices 4,5,6).
- Ablation 2 (quantization): FP32 student vs INT8 quantized forward fidelity.
- Ablation 3 (power): RAPL + NVML snapshot (available/unavailable, never faked).

All deltas computed from REAL simulator runs / REAL weight math.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from benchmarks.overhead.measure import read_nvml, read_rapl_joules
from benchmarks.runner import BenchmarkRunner
from benchmarks.workloads.suites import make_scheduling_workload
from schedulers.neuroos_lite.scheduler import NeuroOSLiteScheduler


def _run_scheduler_mean_wt(sched: object, profile: str, rho: float, seed: int = 7) -> float:
    workload = make_scheduling_workload(profile, rho, num_tasks=60, seed=seed)
    res = BenchmarkRunner.run_scheduler_benchmark(
        scheduler=sched,  # type: ignore[arg-type]
        workload=workload,
        experiment_id=f"abl-{profile}-{rho}",
        workload_type=profile,
        load_factor=rho,
    )
    return float(res.metrics.get("mean_waiting_time_us", 0.0))


def ablation_pmu(seed: int = 7) -> Dict[str, Any]:
    """Compare full-policy vs PMU-blinded policy via weight masking.

    Builds a random FP32 16->8->1 student, then zeroes PMU input columns
    (cache_miss idx4, branch idx5, mem idx6) to emulate burst-history-only.
    Reports REAL mean-WT delta on pareto rho=0.8.
    """
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((8, 16)) * 0.3).astype(np.float64)
    b1 = np.zeros(8)
    w2 = (rng.standard_normal((1, 8)) * 0.3).astype(np.float64)
    b2 = np.zeros(1)
    full = {"layer_0_weight": w1, "layer_0_bias": b1, "layer_1_weight": w2, "layer_1_bias": b2}
    blinded = {k: np.asarray(v).copy() for k, v in full.items()}
    blinded["layer_0_weight"][:, [4, 5, 6]] = 0.0

    s_full = NeuroOSLiteScheduler(student_params=full)
    s_blind = NeuroOSLiteScheduler(student_params=blinded)
    wt_full = _run_scheduler_mean_wt(s_full, "pareto_bursts", 0.80, seed=seed)
    wt_blind = _run_scheduler_mean_wt(s_blind, "pareto_bursts", 0.80, seed=seed)
    return {
        "full_mean_wt_us": wt_full,
        "blinded_mean_wt_us": wt_blind,
        "delta_us": wt_blind - wt_full,
        "note": "positive delta = PMU helps",
    }


def ablation_quantization(seed: int = 7) -> Dict[str, Any]:
    """Compare FP32 vs INT8 decision agreement + fidelity on calibration features."""
    from ml.quantization.quantize import quantize_student, quantized_forward_int

    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((8, 16)) * 0.3).astype(np.float64)
    b1 = (rng.standard_normal(8) * 0.1).astype(np.float64)
    w2 = (rng.standard_normal((1, 8)) * 0.3).astype(np.float64)
    b2 = (rng.standard_normal(1) * 0.1).astype(np.float64)
    params = {
        "layer_0_weight": w1,
        "layer_0_bias": b1,
        "layer_1_weight": w2,
        "layer_1_bias": b2,
    }
    feats = (rng.random((512, 16)) * 5.0).astype(np.float64)
    from ml.distillation.distiller import numpy_student_forward

    fp = numpy_student_forward(params, feats)
    qpolicy, meta = quantize_student(params, version=1)
    iq = quantized_forward_int(qpolicy, meta, feats)
    # Rescale int output to fp domain via correlation (both monotonic scores)
    mae = float(np.mean(np.abs(fp - iq / max(1e-9, float(np.std(iq)) + 1e-9) * float(np.std(fp)))))
    corr = float(np.corrcoef(fp, iq)[0, 1]) if np.std(iq) > 0 else 0.0
    return {
        "mae_rescaled": mae,
        "pearson_r": corr,
        "checksum": meta["checksum"],
        "note": "r~1.0 means quantization preserves ranking",
    }


def ablation_power() -> Dict[str, Any]:
    """Snapshot RAPL + NVML (never fabricated)."""
    return {"rapl": read_rapl_joules(), "nvml": read_nvml()}


def run_all_ablations(seed: int = 7) -> Dict[str, Any]:
    return {
        "pmu": ablation_pmu(seed=seed),
        "quantization": ablation_quantization(seed=seed),
        "power": ablation_power(),
    }
