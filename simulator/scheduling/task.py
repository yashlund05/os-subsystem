"""Task state representation for discrete-event scheduling simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskState(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"


@dataclass
class SimulatedTask:
    """
    Represents a task within the uniprocessor scheduling simulator.
    Mirrors `struct task_descriptor` from kernel/schedulers/include/scheduler_interface.h.
    """

    pid: int
    arrival_time_us: int
    total_burst_us: int
    executed_burst_us: int = 0
    remaining_burst_us: int = field(init=False)
    last_dispatch_time_us: int = 0
    waiting_time_us: int = 0
    first_dispatch_time_us: Optional[int] = None
    completion_time_us: Optional[int] = None
    context_switches: int = 0
    priority_level: int = 0
    sleep_time_us: int = 0
    state: TaskState = TaskState.READY

    # Hardware PMU simulated metrics (functions of elapsed execution time only)
    cache_miss_rate: float = 0.02
    branch_mispred_rate: float = 0.01
    memory_footprint_kb: int = 4096

    @property
    def cache_misses(self) -> int:
        """Returns PMU cache misses accumulated strictly during elapsed execution."""
        return int(self.executed_burst_us * self.cache_miss_rate)

    @property
    def branch_mispredictions(self) -> int:
        """Returns PMU branch mispredictions accumulated strictly during elapsed execution."""
        return int(self.executed_burst_us * self.branch_mispred_rate)

    def __post_init__(self) -> None:
        self.remaining_burst_us = self.total_burst_us

    @property
    def turnaround_time_us(self) -> Optional[int]:
        if self.completion_time_us is None:
            return None
        return self.completion_time_us - self.arrival_time_us

    @property
    def normalized_turnaround_time(self) -> Optional[float]:
        tat = self.turnaround_time_us
        if tat is None or self.total_burst_us == 0:
            return None
        return tat / self.total_burst_us

    @property
    def response_time_us(self) -> Optional[int]:
        if self.first_dispatch_time_us is None:
            return None
        return self.first_dispatch_time_us - self.arrival_time_us
