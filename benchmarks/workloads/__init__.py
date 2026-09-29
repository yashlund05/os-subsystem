"""Workload suites package."""

from benchmarks.workloads.suites import (
    LOAD_FACTORS_10,
    MEMORY_TRACES,
    SCHEDULING_PROFILES,
    make_memory_trace,
    make_scheduling_workload,
)

__all__ = [
    "LOAD_FACTORS_10",
    "SCHEDULING_PROFILES",
    "MEMORY_TRACES",
    "make_scheduling_workload",
    "make_memory_trace",
]
