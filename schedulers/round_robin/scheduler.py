"""Round Robin (RR) Preemptive Scheduler with Configurable Quantum."""

from collections import deque
from typing import Deque, List, Optional, Tuple

from schedulers.base import BaseScheduler
from simulator.scheduling.task import SimulatedTask


class RoundRobinScheduler(BaseScheduler):
    """
    Round Robin (RR) Preemptive CPU Scheduler.
    - Category: Preemptive.
    - Queueing: Circular FIFO queue.
    - Allocation: Fixed time quantum `quantum_us` (default 5000 us = 5 ms).
    """

    def __init__(self, quantum_us: int = 5000) -> None:
        super().__init__(name=f"RoundRobin(q={quantum_us}us)", is_preemptive=True)
        self.quantum_us = max(1, quantum_us)
        self.queue: Deque[SimulatedTask] = deque()

    def add_task(self, task: SimulatedTask, current_time_us: int) -> None:
        self.queue.append(task)

    def pick_next_task(self, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        if not self.queue:
            return None, 0
        task = self.queue.popleft()
        allocated_quantum = min(self.quantum_us, task.remaining_burst_us)
        return task, allocated_quantum

    def has_runnable_tasks(self) -> bool:
        return len(self.queue) > 0

    def get_queue_depth(self) -> int:
        return len(self.queue)

    def get_runnable_tasks(self) -> List[SimulatedTask]:
        return list(self.queue)
