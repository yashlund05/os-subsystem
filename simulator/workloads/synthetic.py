"""Synthetic workload generation using Pareto distributions and Poisson arrival processes."""

import random
from typing import List
from simulator.scheduling.task import SimulatedTask


class SyntheticWorkloadGenerator:
    """
    Generates synthetic task workloads per docs/Experimental-Protocol.md:
    - Burst sizes via Pareto distribution (alpha in [1.1, 1.8])
    - Inter-arrival times via Exponential distribution (Poisson arrival process)
    """
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_pareto_bursts(
        self,
        num_tasks: int,
        alpha: float = 1.3,
        min_burst_us: int = 100,
        mean_inter_arrival_us: float = 500.0
    ) -> List[SimulatedTask]:
        """
        Generates tasks with Pareto-distributed CPU bursts and Poisson arrival timestamps.
        """
        tasks: List[SimulatedTask] = []
        current_time_us = 0

        for pid in range(1, num_tasks + 1):
            # Exponential inter-arrival duration (Poisson process)
            inter_arrival = int(self.rng.expovariate(1.0 / mean_inter_arrival_us))
            current_time_us += inter_arrival

            # Pareto burst duration: x = x_m / (U^(1/alpha))
            u = self.rng.random()
            burst = int(min_burst_us / (u ** (1.0 / alpha)))
            burst = max(min_burst_us, min(burst, 10_000_000)) # Clamped

            tasks.append(
                SimulatedTask(
                    pid=pid,
                    arrival_time_us=current_time_us,
                    total_burst_us=burst
                )
            )

        return tasks
