"""Synthetic workload generation using Pareto distributions and Poisson arrival processes."""

import random
from typing import List

from simulator.scheduling.task import SimulatedTask


class SyntheticWorkloadGenerator:
    """
    Generates synthetic task workloads per docs/Experimental-Protocol.md:
    - Burst sizes via Pareto distribution (alpha in [1.1, 1.8])
    - Inter-arrival times via Exponential distribution (Poisson arrival process)
    - Offered load factors rho in [0.10, 0.98]
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate_pareto_bursts(
        self, num_tasks: int, alpha: float = 1.3, min_burst_us: int = 100, load_factor: float = 0.80
    ) -> List[SimulatedTask]:
        """
        Generates tasks with Pareto-distributed CPU bursts and Poisson arrival timestamps.
        Offered load factor: rho = lambda * E[b]
        => mean_inter_arrival = 1 / lambda = E[b] / rho
        """
        if alpha <= 1.0:
            raise ValueError("Pareto alpha must be strictly greater than 1.0 for finite mean")

        # Theoretical expected burst for Pareto: E[b] = alpha * min_burst / (alpha - 1)
        expected_burst = (alpha * min_burst_us) / (alpha - 1.0)
        mean_inter_arrival_us = expected_burst / max(0.01, min(0.99, load_factor))

        tasks: List[SimulatedTask] = []
        current_time_us = 0

        for pid in range(1, num_tasks + 1):
            # Exponential inter-arrival duration (Poisson process)
            inter_arrival = max(1, int(self.rng.expovariate(1.0 / mean_inter_arrival_us)))
            current_time_us += inter_arrival

            # Pareto burst duration: x = x_m / (U^(1/alpha))
            u = self.rng.random()
            burst = int(min_burst_us / (u ** (1.0 / alpha)))
            burst = max(min_burst_us, min(burst, 10_000_000))  # Clamped to 10s max

            # Simulated hardware PMU counter deltas
            cache_misses = int(burst * self.rng.uniform(0.01, 0.05))
            branch_mispredictions = int(burst * self.rng.uniform(0.005, 0.02))
            footprint_kb = int(self.rng.uniform(512, 16384))

            tasks.append(
                SimulatedTask(
                    pid=pid,
                    arrival_time_us=current_time_us,
                    total_burst_us=burst,
                    cache_misses=cache_misses,
                    branch_mispredictions=branch_mispredictions,
                    memory_footprint_kb=footprint_kb,
                )
            )

        return tasks
