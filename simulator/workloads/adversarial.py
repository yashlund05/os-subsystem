"""Adversarial and pathological workload generators for stress-testing OS subsystems."""

from dataclasses import dataclass
from enum import Enum
from typing import List

from simulator.scheduling.task import SimulatedTask


class AdversarialWorkloadType(str, Enum):
    CONVOY_TRIGGER = "CONVOY_TRIGGER"
    STARVATION_CASCADE = "STARVATION_CASCADE"
    MEMORY_ODD_EVEN_CHURN = "MEMORY_ODD_EVEN_CHURN"


@dataclass
class MemoryEvent:
    event_id: int
    is_alloc: bool
    size_bytes: int
    predicted_lifetime_us: int
    task_pid: int
    target_alloc_id: int = 0


class AdversarialWorkloadGenerator:
    """
    Generates pathological edge-case workloads per docs/Experimental-Protocol.md.
    """

    @staticmethod
    def create_convoy_workload(
        num_short_jobs: int = 100, long_burst_us: int = 100000, short_burst_us: int = 10
    ) -> List[SimulatedTask]:
        """
        Creates a pathological convoy trigger:
        One massive job arrives at t=0, immediately followed by a cascade of tiny interactive jobs.
        """
        tasks: List[SimulatedTask] = [
            SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=long_burst_us)
        ]

        for i in range(2, num_short_jobs + 2):
            tasks.append(
                SimulatedTask(
                    pid=i,
                    arrival_time_us=i - 1,  # Arrive immediately behind the long job
                    total_burst_us=short_burst_us,
                )
            )

        return tasks

    @staticmethod
    def create_memory_fragmentation_churn(
        num_pairs: int = 50, small_size_bytes: int = 4096, large_size_bytes: int = 65536
    ) -> List[MemoryEvent]:
        """
        Creates alternating odd-even allocations, then frees every second block (odd blocks),
        producing maximum external fragmentation holes and slivers.
        """
        events: List[MemoryEvent] = []
        event_id = 1

        # Phase 1: Interleaved allocations of small and large blocks
        for i in range(1, num_pairs + 1):
            # Odd block (small)
            events.append(
                MemoryEvent(
                    event_id=event_id,
                    is_alloc=True,
                    size_bytes=small_size_bytes,
                    predicted_lifetime_us=1000,
                    task_pid=i * 2 - 1,
                )
            )
            event_id += 1

            # Even block (large)
            events.append(
                MemoryEvent(
                    event_id=event_id,
                    is_alloc=True,
                    size_bytes=large_size_bytes,
                    predicted_lifetime_us=100000,
                    task_pid=i * 2,
                )
            )
            event_id += 1

        # Phase 2: Free all small odd blocks to leave holes between large blocks
        for i in range(1, num_pairs + 1):
            odd_alloc_event_id = (i - 1) * 2 + 1
            events.append(
                MemoryEvent(
                    event_id=event_id,
                    is_alloc=False,
                    size_bytes=small_size_bytes,
                    predicted_lifetime_us=0,
                    task_pid=i * 2 - 1,
                    target_alloc_id=odd_alloc_event_id,
                )
            )
            event_id += 1

        return events
