"""Full scheduling benchmark matrix (Phase 4 Week 7).

Executes 7 algorithms x 5 profiles x 10 load factors = 350 REAL simulator runs.
Metrics per docs/Metrics.md + docs/Schema.md. No fabricated numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List

from allocators.base import BaseAllocator  # noqa: F401 (re-export symmetry)
from benchmarks.result_schema import BenchmarkResult
from benchmarks.runner import BenchmarkRunner
from benchmarks.workloads.suites import (
    LOAD_FACTORS_10,
    SCHEDULING_PROFILES,
    make_scheduling_workload,
)
from schedulers.neuroos_lite.scheduler import NeuroOSLiteScheduler


def build_schedulers() -> Dict[str, Any]:
    from schedulers.fcfs.scheduler import FCFSScheduler
    from schedulers.mlfq.scheduler import MLFQScheduler
    from schedulers.round_robin.scheduler import RoundRobinScheduler
    from schedulers.sjf.scheduler import SJFScheduler
    from schedulers.srtf.scheduler import SRTFScheduler

    return {
        "FCFS": FCFSScheduler(),
        "SJF": SJFScheduler(),
        "SRTF": SRTFScheduler(),
        "RR_5ms": RoundRobinScheduler(quantum_us=5000),
        "RR_20ms": RoundRobinScheduler(quantum_us=20000),
        "MLFQ": MLFQScheduler(),
        "NeuroOS-Lite": NeuroOSLiteScheduler(),
    }


def run_scheduling_matrix(
    num_tasks: int = 100,
    seed: int = 42,
    profiles: List[str] | None = None,
    load_factors: List[float] | None = None,
) -> List[BenchmarkResult]:
    profiles = profiles or SCHEDULING_PROFILES
    load_factors = load_factors or LOAD_FACTORS_10
    schedulers = build_schedulers()
    results: List[BenchmarkResult] = []
    for algo_name, sched in schedulers.items():
        for profile in profiles:
            for rho in load_factors:
                workload = make_scheduling_workload(
                    profile, load_factor=rho, num_tasks=num_tasks, seed=seed
                )
                exp_id = f"sched-{algo_name}-{profile}-rho{rho}-s{seed}"
                res = BenchmarkRunner.run_scheduler_benchmark(
                    scheduler=sched,
                    workload=workload,
                    experiment_id=exp_id,
                    workload_type=profile,
                    load_factor=rho,
                )
                # Attach guardrail trips for learned scheduler
                trips = int(getattr(sched, "fallback_trips", 0))
                res.guardrail_fallback_trips = trips
                # Reset stateful learned scheduler between runs
                if isinstance(sched, NeuroOSLiteScheduler):
                    sched.queue.clear()
                    sched.fallback_trips = 0
                    sched.burst_estimator.reset()
                results.append(res)
    return results
