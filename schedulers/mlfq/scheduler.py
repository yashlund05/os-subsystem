"""Multi-Level Feedback Queue (MLFQ) Preemptive Scheduler."""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from schedulers.base import BaseScheduler
from simulator.scheduling.task import SimulatedTask


class MLFQScheduler(BaseScheduler):
    """
    Multi-Level Feedback Queue (MLFQ) Scheduler.
    - Category: Preemptive multi-tier feedback queues.
    - Structure: K priority levels (0 is highest, K-1 is lowest).
    - Quantum scaling: Geometric scaling per tier: q_k = base_quantum_us * (2^k).
    - Priority demotion: Tasks using their full quantum are demoted to the next lower priority queue.
    - Priority boost: Periodic global boost (every boost_interval_us) moves all tasks to tier 0 to prevent starvation.
    """

    def __init__(
        self, num_levels: int = 3, base_quantum_us: int = 5000, boost_interval_us: int = 50000
    ) -> None:
        super().__init__(name="MLFQ", is_preemptive=True)
        self.num_levels = max(1, num_levels)
        self.base_quantum_us = base_quantum_us
        self.boost_interval_us = boost_interval_us

        # Quanta per level: e.g. [5000, 10000, 20000]
        self.quanta: List[int] = [base_quantum_us * (2**k) for k in range(self.num_levels)]

        # Queues per level
        self.queues: List[Deque[SimulatedTask]] = [deque() for _ in range(self.num_levels)]

        # Track task priority levels
        self.task_levels: Dict[int, int] = {}
        self.last_boost_time_us: int = 0

    def add_task(self, task: SimulatedTask, current_time_us: int) -> None:
        self._check_priority_boost(current_time_us)
        level = self.task_levels.get(task.pid, 0)
        self.task_levels[task.pid] = level
        task.priority_level = level
        self.queues[level].append(task)

    def on_task_preempted(self, task: SimulatedTask, current_time_us: int) -> None:
        """Demote task priority upon quantum expiration."""
        self._check_priority_boost(current_time_us)
        current_level = self.task_levels.get(task.pid, 0)
        new_level = min(self.num_levels - 1, current_level + 1)
        self.task_levels[task.pid] = new_level
        task.priority_level = new_level
        self.queues[new_level].append(task)

    def pick_next_task(self, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        self._check_priority_boost(current_time_us)
        for level in range(self.num_levels):
            if self.queues[level]:
                task = self.queues[level].popleft()
                allocated_quantum = min(self.quanta[level], task.remaining_burst_us)
                return task, allocated_quantum
        return None, 0

    def on_tick(self, current_time_us: int, elapsed_us: int) -> None:
        self._check_priority_boost(current_time_us)

    def _check_priority_boost(self, current_time_us: int) -> None:
        if (
            self.boost_interval_us > 0
            and (current_time_us - self.last_boost_time_us) >= self.boost_interval_us
        ):
            self.last_boost_time_us = current_time_us
            # Move all tasks across all levels to level 0
            for level in range(1, self.num_levels):
                while self.queues[level]:
                    task = self.queues[level].popleft()
                    self.task_levels[task.pid] = 0
                    task.priority_level = 0
                    self.queues[0].append(task)

    def has_runnable_tasks(self) -> bool:
        return any(len(q) > 0 for q in self.queues)

    def get_queue_depth(self) -> int:
        return sum(len(q) for q in self.queues)

    def get_runnable_tasks(self) -> List[SimulatedTask]:
        tasks: List[SimulatedTask] = []
        for q in self.queues:
            tasks.extend(list(q))
        return tasks
