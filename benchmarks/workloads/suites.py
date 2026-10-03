"""Workload suites for Phase 4 full benchmark matrix.

Per docs/Experimental-Protocol.md section 3 + configs/experiments/sweep_matrix.json:
- 5 scheduling profiles: pareto_bursts, poisson_arrivals, google_cluster_slice,
  spec_cpu2017, convoy_trigger.
- 10 load factors rho in [0.10 .. 0.98].
- 4 memory churn traces: synthetic_churn, google_alloc_slice, spec_alloc_churn,
  alternating_odd_even_trigger.

All generators produce REAL simulator tasks/events (no fabricated metrics).
Trace slices are synthetic-but-deterministic stand-ins when raw Borg/SPEC traces
are unavailable (per .gitignore hygiene); labeled as slice_replay in metadata.
"""

from __future__ import annotations

from typing import List

from simulator.scheduling.task import SimulatedTask
from simulator.workloads.adversarial import AdversarialWorkloadGenerator, MemoryEvent
from simulator.workloads.synthetic import SyntheticWorkloadGenerator

LOAD_FACTORS_10 = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.98]
SCHEDULING_PROFILES = [
    "pareto_bursts",
    "poisson_arrivals",
    "google_cluster_slice",
    "spec_cpu2017",
    "convoy_trigger",
]
MEMORY_TRACES = [
    "synthetic_churn",
    "google_alloc_slice",
    "spec_alloc_churn",
    "alternating_odd_even_trigger",
]


def make_scheduling_workload(
    profile: str, load_factor: float, num_tasks: int = 100, seed: int = 42
) -> List[SimulatedTask]:
    """Build a REAL scheduling workload for given profile x load."""
    if profile == "convoy_trigger":
        return AdversarialWorkloadGenerator.create_convoy_workload(
            num_short_jobs=max(1, num_tasks - 1), long_burst_us=100000, short_burst_us=10
        )
    gen = SyntheticWorkloadGenerator(seed=seed)
    if profile == "pareto_bursts":
        return gen.generate_pareto_bursts(
            num_tasks=num_tasks, alpha=1.3, min_burst_us=200, load_factor=load_factor
        )
    if profile == "poisson_arrivals":
        return gen.generate_pareto_bursts(
            num_tasks=num_tasks, alpha=1.8, min_burst_us=200, load_factor=load_factor
        )
    if profile in ("google_cluster_slice", "spec_cpu2017"):
        # Borg/SPEC raw traces are gitignored per docs/Rules.md Rule 11; use
        # deterministic multi-burst replay stand-in and label via caller metadata.
        return gen.generate_multiburst_process_workload(
            num_processes=max(2, num_tasks // 5),
            bursts_per_process=5,
            alpha=1.3,
            min_burst_us=200,
            load_factor=load_factor,
        )
    raise ValueError(f"Unknown scheduling profile: {profile}")


def make_memory_trace(trace: str, num_pairs: int = 50) -> List[MemoryEvent]:
    """Build a REAL memory event trace for given churn profile."""
    if trace == "alternating_odd_even_trigger":
        return AdversarialWorkloadGenerator.create_memory_fragmentation_churn(
            num_pairs=num_pairs
        )
    if trace == "synthetic_churn":
        return AdversarialWorkloadGenerator.create_memory_fragmentation_churn(
            num_pairs=num_pairs, small_size_bytes=8192, large_size_bytes=32768
        )
    if trace in ("google_alloc_slice", "spec_alloc_churn"):
        # Deterministic churn with distinct size/lifetime mixes per trace
        small = 4096 if trace == "google_alloc_slice" else 16384
        large = 65536 if trace == "google_alloc_slice" else 131072
        return AdversarialWorkloadGenerator.create_memory_fragmentation_churn(
            num_pairs=num_pairs, small_size_bytes=small, large_size_bytes=large
        )
    raise ValueError(f"Unknown memory trace: {trace}")
