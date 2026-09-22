"""Automated benchmark harness for evaluating scheduling and memory allocators."""

from typing import Dict, List

from allocators.base import BaseAllocator, MemoryHandle
from benchmarks.result_schema import BenchmarkResult, SubsystemType
from schedulers.base import BaseScheduler
from simulator.scheduling.engine import SchedulingSimulationEngine
from simulator.scheduling.task import SimulatedTask
from simulator.workloads.adversarial import MemoryEvent


class BenchmarkRunner:
    """
    Automated benchmark runner executing workload matrices across algorithms.
    Produces structured BenchmarkResult records conforming to docs/Schema.md.
    """

    @staticmethod
    def run_scheduler_benchmark(
        scheduler: BaseScheduler,
        workload: List[SimulatedTask],
        experiment_id: str,
        workload_type: str = "synthetic_pareto",
        load_factor: float = 0.80,
        context_switch_overhead_us: int = 2,
    ) -> BenchmarkResult:
        engine = SchedulingSimulationEngine(
            scheduler=scheduler, context_switch_overhead_us=context_switch_overhead_us
        )
        _, metrics = engine.run(workload)

        metric_dict: Dict[str, float] = {
            "mean_turnaround_time_us": metrics.mean_turnaround_time_us,
            "mean_normalized_turnaround_time": metrics.mean_normalized_turnaround_time,
            "mean_waiting_time_us": metrics.mean_waiting_time_us,
            "p95_waiting_time_us": metrics.p95_waiting_time_us,
            "p99_waiting_time_us": metrics.p99_waiting_time_us,
            "p999_waiting_time_us": metrics.p999_waiting_time_us,
            "total_context_switches": float(metrics.total_context_switches),
            "cpu_utilization": metrics.cpu_utilization,
            "overhead_ratio": metrics.overhead_ratio,
            "total_makespan_us": float(metrics.total_makespan_us),
        }

        return BenchmarkResult(
            experiment_id=experiment_id,
            subsystem=SubsystemType.SCHEDULING,
            algorithm=scheduler.name,
            workload_type=workload_type,
            load_factor=load_factor,
            metrics=metric_dict,
            guardrail_fallback_trips=0,
        )

    @staticmethod
    def run_memory_benchmark(
        allocator: BaseAllocator,
        events: List[MemoryEvent],
        experiment_id: str,
        workload_type: str = "alternating_odd_even",
    ) -> BenchmarkResult:
        allocator.reset()
        active_handles: Dict[int, MemoryHandle] = {}

        for event in events:
            if event.is_alloc:
                handle = allocator.allocate(
                    request_id=event.event_id,
                    task_pid=event.task_pid,
                    size_bytes=event.size_bytes,
                    predicted_lifetime_us=event.predicted_lifetime_us,
                )
                if handle is not None:
                    active_handles[event.event_id] = handle
            else:
                target_handle = active_handles.pop(event.target_alloc_id, None)
                if target_handle is not None:
                    allocator.deallocate(target_handle)

        final_metrics = allocator.get_metrics()
        metric_dict: Dict[str, float] = {
            "external_fragmentation": final_metrics.external_fragmentation,
            "internal_fragmentation": final_metrics.internal_fragmentation,
            "buffer_utilization": final_metrics.buffer_utilization,
            "allocated_bytes": float(final_metrics.allocated_bytes),
            "free_bytes": float(final_metrics.free_bytes),
            "max_free_block_bytes": float(final_metrics.max_free_block_bytes),
            "total_alloc_requests": float(final_metrics.total_alloc_requests),
            "failed_alloc_requests": float(final_metrics.failed_alloc_requests),
            "total_dealloc_requests": float(final_metrics.total_dealloc_requests),
        }

        return BenchmarkResult(
            experiment_id=experiment_id,
            subsystem=SubsystemType.MEMORY,
            algorithm=allocator.name,
            workload_type=workload_type,
            load_factor=1.0,
            metrics=metric_dict,
            guardrail_fallback_trips=0,
        )
