"""Shortest Remaining Time First (SRTF) Preemptive Scheduler."""

from typing import List, Optional, Tuple

from schedulers.base import BaseScheduler
from simulator.scheduling.task import SimulatedTask


class SRTFScheduler(BaseScheduler):
    """
    Shortest Remaining Time First (SRTF) Preemptive CPU Scheduler.
    - Category: Preemptive.
    - Selection: Task with shortest remaining burst duration.
    - Preemption: Newly arrived tasks preempt the running task if their remaining burst is strictly less.
    """

    def __init__(self) -> None:
        super().__init__(name="SRTF", is_preemptive=True)
        self.ready_tasks: List[SimulatedTask] = []

    def add_task(self, task: SimulatedTask, current_time_us: int) -> None:
        self.ready_tasks.append(task)
        self._sort_ready_queue()

    def _sort_ready_queue(self) -> None:
        # Sort by remaining burst ascending, then arrival time, then PID
        self.ready_tasks.sort(key=lambda t: (t.remaining_burst_us, t.arrival_time_us, t.pid))

    def pick_next_task(self, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        if not self.ready_tasks:
            return None, 0
        self._sort_ready_queue()
        task = self.ready_tasks.pop(0)
        # Returns task and remaining burst; engine will preempt if shorter task arrives
        return task, task.remaining_burst_us

    def should_preempt(self, running_task: SimulatedTask, candidate_task: SimulatedTask) -> bool:
        """Determines if a candidate arrival should preempt the currently executing task."""
        return candidate_task.remaining_burst_us < running_task.remaining_burst_us

    def has_runnable_tasks(self) -> bool:
        return len(self.ready_tasks) > 0

    def get_queue_depth(self) -> int:
        return len(self.ready_tasks)

    def get_runnable_tasks(self) -> List[SimulatedTask]:
        return list(self.ready_tasks)
