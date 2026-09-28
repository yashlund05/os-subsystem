"""Synthetic workload generation using Pareto distributions and Poisson arrival processes."""

import math
import random
from typing import Dict, List

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

            # Simulated hardware PMU event rates (per microsecond of execution, independent of burst)
            cache_miss_rate = self.rng.uniform(0.01, 0.05)
            branch_mispred_rate = self.rng.uniform(0.005, 0.02)
            footprint_kb = int(self.rng.uniform(512, 16384))

            tasks.append(
                SimulatedTask(
                    pid=pid,
                    arrival_time_us=current_time_us,
                    total_burst_us=burst,
                    cache_miss_rate=cache_miss_rate,
                    branch_mispred_rate=branch_mispred_rate,
                    memory_footprint_kb=footprint_kb,
                )
            )

        return tasks

    def generate_multiburst_process_workload(
        self,
        num_processes: int = 10,
        bursts_per_process: int = 5,
        alpha: float = 1.3,
        min_burst_us: int = 200,
        load_factor: float = 0.8,
        within_process_sigma: float = 0.30,
    ) -> List[SimulatedTask]:
        """
        Generates realistic multi-burst process workload where PIDs repeat across bursts.
        Each process has a characteristic mean burst drawn from Pareto, and successive bursts
        are lognormally distributed around that process mean.
        Enables genuine, imperfect EMA prediction without ground-truth oracle access.
        """
        # 1. Establish per-process baseline characteristics
        process_means: Dict[int, float] = {}
        process_cache_rate: Dict[int, float] = {}
        process_branch_rate: Dict[int, float] = {}
        process_mem: Dict[int, int] = {}

        for pid in range(1, num_processes + 1):
            u = self.rng.random()
            mu = float(min_burst_us / (u ** (1.0 / alpha)))
            process_means[pid] = max(min_burst_us, min(mu, 500000))
            process_cache_rate[pid] = self.rng.uniform(0.01, 0.05)
            process_branch_rate[pid] = self.rng.uniform(0.005, 0.02)
            process_mem[pid] = int(self.rng.uniform(512, 16384))

        # Overall expected burst across all processes
        avg_burst = float(sum(process_means.values()) / max(1, len(process_means)))
        mean_inter_arrival_us = avg_burst / max(0.01, min(0.99, load_factor))

        tasks: List[SimulatedTask] = []
        current_time_us = 0
        total_tasks = num_processes * bursts_per_process

        # Interleave burst arrivals across processes
        for burst_idx in range(bursts_per_process):
            pids = list(range(1, num_processes + 1))
            self.rng.shuffle(pids)
            for pid in pids:
                inter_arrival = max(1, int(self.rng.expovariate(1.0 / mean_inter_arrival_us)))
                current_time_us += inter_arrival

                # Burst length drawn from per-process lognormal distribution around process mean
                factor = float(math.exp(self.rng.gauss(0.0, within_process_sigma)))
                b = int(process_means[pid] * factor)
                burst = max(min_burst_us, min(b, 10_000_000))

                tasks.append(
                    SimulatedTask(
                        pid=pid,
                        arrival_time_us=current_time_us,
                        total_burst_us=burst,
                        cache_miss_rate=process_cache_rate[pid],
                        branch_mispred_rate=process_branch_rate[pid],
                        memory_footprint_kb=process_mem[pid],
                    )
                )

        return tasks
