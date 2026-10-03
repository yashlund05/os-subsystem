"""Phase 6 Multi-Core SMP, NUMA Allocation, and Online Drift Benchmark Harness.

Compares:
1. Scaling: 1, 2, 4, 8 CPU cores (makespan, turnaround time, utilization).
2. NUMA Memory Allocation: Local node hit rate vs remote interconnect hops.
3. Online Drift Self-Correction: Fallback trip reduction under distribution shift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from allocators.numa_lifetime.numa_allocator import NUMALifetimeAllocator
from schedulers.smp_neuroos.smp_scheduler import SMPNeuroOSLiteScheduler
from simulator.scheduling.smp_engine import SMPSchedulingSimulationEngine
from simulator.scheduling.task import SimulatedTask


def generate_benchmark_workload(
    num_tasks: int = 100, burst_scale_us: int = 2500
) -> List[SimulatedTask]:
    tasks = []
    for i in range(num_tasks):
        arrival = i * 150
        burst = burst_scale_us + (i % 7) * 800
        tasks.append(SimulatedTask(pid=i + 1, arrival_time_us=arrival, total_burst_us=burst))
    return tasks


def run_scaling_benchmark() -> Dict[str, Dict[str, float]]:
    results = {}
    workload = generate_benchmark_workload(num_tasks=80)

    for num_cpus in [1, 2, 4, 8]:
        num_nodes = 1 if num_cpus <= 2 else 2
        sched = SMPNeuroOSLiteScheduler(num_cpus=num_cpus, num_numa_nodes=num_nodes)
        engine = SMPSchedulingSimulationEngine(sched, num_cpus=num_cpus, num_numa_nodes=num_nodes)
        _, metrics = engine.run(workload)

        results[f"{num_cpus}_core"] = {
            "num_cpus": float(num_cpus),
            "makespan_us": float(metrics.total_makespan_us),
            "mean_turnaround_us": float(metrics.mean_turnaround_time_us),
            "mean_waiting_us": float(metrics.mean_waiting_time_us),
            "utilization": float(metrics.overall_cpu_utilization),
            "migrations": float(metrics.total_migrations),
        }
    return results


def run_numa_allocation_benchmark() -> Dict[str, float]:
    allocator = NUMALifetimeAllocator(total_heap_bytes=16 * 1024 * 1024, num_nodes=2)
    # Simulate multi-threaded allocation stream with CPU thread affinity
    for i in range(200):
        pref_node = 0 if (i % 4) < 2 else 1
        allocator.allocate(
            request_id=i + 1,
            task_pid=(i % 16) + 1,
            size_bytes=1024 + (i % 5) * 512,
            predicted_lifetime_us=5000 * (1 + (i % 4)),
            preferred_node=pref_node,
        )

    metrics = allocator.compute_metrics()
    total_local = allocator.local_node_allocations
    total_remote = allocator.remote_node_allocations
    total_reqs = total_local + total_remote + allocator.fallback_trips

    return {
        "local_node_hit_rate": float(total_local) / max(1.0, float(total_reqs)),
        "remote_node_rate": float(total_remote) / max(1.0, float(total_reqs)),
        "fallback_trips": float(allocator.fallback_trips),
        "external_fragmentation": float(metrics.external_fragmentation),
    }


def run_online_drift_benchmark() -> Dict[str, float]:
    # Non-stationary workload: sudden 4x burst inflation simulating unexpected workload phase shift
    shifted_workload = [
        SimulatedTask(pid=i + 1, arrival_time_us=i * 100, total_burst_us=10000 if i >= 20 else 1500)
        for i in range(60)
    ]

    # Baseline scheduler without online drift correction
    sched_no_corr = SMPNeuroOSLiteScheduler(num_cpus=2, enable_online_drift_correction=False)
    engine_no_corr = SMPSchedulingSimulationEngine(sched_no_corr, num_cpus=2)
    engine_no_corr.run(shifted_workload)

    # Phase 6 scheduler with online drift correction
    sched_with_corr = SMPNeuroOSLiteScheduler(num_cpus=2, enable_online_drift_correction=True)
    engine_with_corr = SMPSchedulingSimulationEngine(sched_with_corr, num_cpus=2)
    engine_with_corr.run(shifted_workload)

    return {
        "trips_without_online_correction": float(sched_no_corr.fallback_trips),
        "trips_with_online_correction": float(sched_with_corr.fallback_trips),
        "trip_reduction_ratio": 1.0
        - (float(sched_with_corr.fallback_trips) / max(1.0, float(sched_no_corr.fallback_trips))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 6 SMP, NUMA, and Online Drift Benchmarks"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/smp/results.json", help="Path to output JSON"
    )
    args = parser.parse_args()

    print("[Phase 6 Benchmark] Running SMP Multi-Core Scaling Sweep...")
    scaling_res = run_scaling_benchmark()

    print("[Phase 6 Benchmark] Running NUMA-Aware Memory Allocation Sweep...")
    numa_res = run_numa_allocation_benchmark()

    print("[Phase 6 Benchmark] Running Online Drift Self-Correction Sweep...")
    drift_res = run_online_drift_benchmark()

    all_results = {
        "smp_scaling": scaling_res,
        "numa_allocation": numa_res,
        "online_drift_correction": drift_res,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Phase 6 Benchmark] Completed successfully. Results written to {out_path}")
    print("\n--- SMP Scaling Summary ---")
    for core, data in scaling_res.items():
        print(
            f"  {core}: Makespan = {data['makespan_us']:.0f} us | Mean WT = {data['mean_waiting_us']:.1f} us | Migrations = {data['migrations']:.0f}"
        )

    print("\n--- NUMA Allocation Summary ---")
    print(f"  Local Node Hit Rate: {numa_res['local_node_hit_rate'] * 100:.1f}%")
    print(f"  External Fragmentation: {numa_res['external_fragmentation'] * 100:.2f}%")

    print("\n--- Online Drift Correction Summary ---")
    print(
        f"  Fallback Trips Without Correction: {drift_res['trips_without_online_correction']:.0f}"
    )
    print(
        f"  Fallback Trips With Online Correction: {drift_res['trips_with_online_correction']:.0f}"
    )
    print(f"  Fallback Trip Reduction: {drift_res['trip_reduction_ratio'] * 100:.1f}%\n")


if __name__ == "__main__":
    main()
