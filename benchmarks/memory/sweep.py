"""Full memory benchmark matrix (Phase 4 Week 7).

Executes 5 allocators x 4 churn traces = 20 REAL heap simulations.
Metrics per docs/Metrics.md (external/internal fragmentation, utilization).
"""

from __future__ import annotations

from typing import Dict, List

from allocators.best_fit.allocator import BestFitAllocator
from allocators.buddy.allocator import BuddyAllocator
from allocators.first_fit.allocator import FirstFitAllocator
from allocators.fixed_partition.allocator import FixedPartitionAllocator
from allocators.lifetime_affinity.allocator import LifetimeAffinityAllocator
from benchmarks.result_schema import BenchmarkResult
from benchmarks.runner import BenchmarkRunner
from benchmarks.workloads.suites import MEMORY_TRACES, make_memory_trace


def build_allocators(heap_bytes: int = 4 * 1024 * 1024) -> Dict[str, object]:
    return {
        "Fixed_MFT": FixedPartitionAllocator(total_heap_bytes=heap_bytes),
        "First_Fit": FirstFitAllocator(total_heap_bytes=heap_bytes),
        "Best_Fit": BestFitAllocator(total_heap_bytes=heap_bytes),
        "Buddy": BuddyAllocator(total_heap_bytes=heap_bytes),
        "NeuroOS-Lite": LifetimeAffinityAllocator(total_heap_bytes=heap_bytes),
    }


def run_memory_matrix(
    heap_bytes: int = 4 * 1024 * 1024,
    traces: List[str] | None = None,
    num_pairs: int = 50,
) -> List[BenchmarkResult]:
    traces = traces or MEMORY_TRACES
    allocators = build_allocators(heap_bytes=heap_bytes)
    results: List[BenchmarkResult] = []
    for algo_name, alloc in allocators.items():
        for trace in traces:
            events = make_memory_trace(trace, num_pairs=num_pairs)
            exp_id = f"mem-{algo_name}-{trace}-p{num_pairs}"
            res = BenchmarkRunner.run_memory_benchmark(
                allocator=alloc,  # type: ignore[arg-type]
                events=events,
                experiment_id=exp_id,
                workload_type=trace,
            )
            trips = int(getattr(alloc, "fallback_trips", 0))
            res.guardrail_fallback_trips = trips
            results.append(res)
    return results
