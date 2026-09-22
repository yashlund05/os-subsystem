from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from simulator.scheduling.task import SimulatedTask


class BaseScheduler(ABC):
    """
    Abstract base class for all CPU schedulers.
    Mirrors `struct scheduler_ops` from kernel/schedulers/include/scheduler_interface.h.
    """

    def __init__(self, name: str, is_preemptive: bool) -> None:
        self.name = name
        self.is_preemptive = is_preemptive

    @abstractmethod
    def add_task(self, task: SimulatedTask, current_time_us: int) -> None:
        """Enqueue a newly arrived or resumed task into the ready queue."""
        pass

    @abstractmethod
    def pick_next_task(self, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        """
        Select the next runnable task from the ready queue.
        Returns:
            Tuple of (selected_task, allocated_quantum_us).
            For non-preemptive schedulers, allocated_quantum_us is the entire remaining burst.
        """
        pass

    def on_tick(self, current_time_us: int, elapsed_us: int) -> None:
        """Periodic timer tick hook (optional override for dynamic quantum or priority accounting)."""
        return None

    def on_task_preempted(self, task: SimulatedTask, current_time_us: int) -> None:
        """Called when a running task's time slice expires or it is preempted by a higher priority task."""
        self.add_task(task, current_time_us)

    def on_task_completion(self, task: SimulatedTask, current_time_us: int) -> None:
        """Called when a task completes its total burst (optional override)."""
        return None

    @abstractmethod
    def has_runnable_tasks(self) -> bool:
        """Returns True if there are tasks waiting in the ready queue."""
        pass

    @abstractmethod
    def get_queue_depth(self) -> int:
        """Returns the number of runnable tasks currently in the ready queue."""
        pass

    @abstractmethod
    def get_runnable_tasks(self) -> List[SimulatedTask]:
        """Returns a list of all currently runnable tasks in queue order."""
        pass
