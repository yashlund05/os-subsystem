"""Shortest Job First (SJF) Non-Preemptive Scheduler."""

import heapq
from typing import Any, List, Optional, Tuple

from schedulers.base import BaseScheduler
from simulator.scheduling.task import SimulatedTask


class SJFScheduler(BaseScheduler):
    """
    Shortest Job First (SJF) CPU Scheduler.
    - Category: Non-preemptive.
    - Selection: Task with shortest total CPU burst (O(log n) min-heap).
    - Tie-breaking: Earlier arrival time, then smaller PID.
    """

    def __init__(self) -> None:
        super().__init__(name="SJF", is_preemptive=False)
        self._heap: List[Tuple[int, int, int, Any]] = []

    def add_task(self, task: SimulatedTask, current_time_us: int) -> None:
        # Tuple key: (total_burst_us, arrival_time_us, pid, task)
        heapq.heappush(self._heap, (task.total_burst_us, task.arrival_time_us, task.pid, task))

    def pick_next_task(self, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        if not self._heap:
            return None, 0
        _, _, _, task = heapq.heappop(self._heap)
        # Non-preemptive: grant entire remaining burst
        return task, task.remaining_burst_us

    def has_runnable_tasks(self) -> bool:
        return len(self._heap) > 0

    def get_queue_depth(self) -> int:
        return len(self._heap)

    def get_runnable_tasks(self) -> List[SimulatedTask]:
        return [item[3] for item in sorted(self._heap)]
