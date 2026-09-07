"""Unit tests for metrics and memory fragmentation formulas."""

import pytest

from benchmarks.result_schema import BenchmarkResult, SubsystemType
from simulator.memory.heap import MemoryBlock, SimulatedHeap


def test_heap_fragmentation_metrics():
    heap = SimulatedHeap(total_size_bytes=1000)

    # Initially 1 free block of 1000 bytes: no external fragmentation
    assert heap.external_fragmentation() == 0.0
    assert heap.internal_fragmentation() == 0.0
    assert heap.buffer_utilization() == 0.0

    # Simulate 3 fragmented free blocks: 100, 200, 100 (total free = 400, max free = 200)
    # External fragmentation = 1 - (200 / 400) = 0.50
    heap.blocks = [
        MemoryBlock(offset=0, size_bytes=100, is_allocated=False),
        MemoryBlock(offset=100, size_bytes=300, is_allocated=True, requested_size_bytes=240),
        MemoryBlock(offset=400, size_bytes=200, is_allocated=False),
        MemoryBlock(offset=600, size_bytes=300, is_allocated=True, requested_size_bytes=260),
        MemoryBlock(offset=900, size_bytes=100, is_allocated=False),
    ]

    assert pytest.approx(heap.external_fragmentation(), 0.001) == 0.50
    # Internal fragmentation: total allocated = 600, total requested = 500. Frag = (600 - 500) / 600 = 100 / 600 = 0.1667
    assert pytest.approx(heap.internal_fragmentation(), 0.001) == 100.0 / 600.0
    # Buffer utilization: 500 / 1000 = 0.50
    assert pytest.approx(heap.buffer_utilization(), 0.001) == 0.50


def test_benchmark_result_serialization():
    res = BenchmarkResult(
        experiment_id="exp-001",
        subsystem=SubsystemType.SCHEDULING,
        algorithm="SRTF",
        workload_type="pareto_alpha_1.3",
        load_factor=0.80,
        metrics={
            "mean_turnaround_time_us": 1420.5,
            "mean_waiting_time_us": 320.1,
            "total_context_switches": 42,
        },
        guardrail_fallback_trips=0,
    )
    d = res.to_dict()
    assert d["experiment_id"] == "exp-001"
    assert d["subsystem"] == "scheduling"
    assert d["algorithm"] == "SRTF"
    assert d["metrics"]["total_context_switches"] == 42
