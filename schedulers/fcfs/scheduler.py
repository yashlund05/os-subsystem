"""First-Come, First-Served (FCFS) Non-Preemptive Scheduler."""

from collections import deque
from typing import Deque, List, Optional, Tuple

from schedulers.base import BaseScheduler
from simulator.scheduling.task import SimulatedTask


class FCFSScheduler(BaseScheduler):
    """
    First-Come, First-Served (FCFS) CPU Scheduler.
    - Category: Non-preemptive.
    - Queueing: FIFO queue (O(1) enqueue and dequeue).
    - Allocation: Runs each task to completion.
    """

    def __init__(self) -> None:
        super().__init__(name="FCFS", is_preemptive=False)
        self.queue: Deque[SimulatedTask] = deque()

    def add_task(self, task: SimulatedTask, current_time_us: int) -> None:
        self.queue.append(task)

    def pick_next_task(self, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        if not self.queue:
            return None, 0
        task = self.queue.popleft()
        # Non-preemptive: grant entire remaining burst
        return task, task.remaining_burst_us

    def has_runnable_tasks(self) -> bool:
        return len(self.queue) > 0

    def get_queue_depth(self) -> int:
        return len(self.queue)

    def get_runnable_tasks(self) -> List[SimulatedTask]:
        return list(self.queue)
