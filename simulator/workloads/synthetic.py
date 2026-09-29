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
        process_burst_means: Dict[int, float] = {}
        process_sleep_means: Dict[int, float] = {}
        process_prios: Dict[int, int] = {}
        process_cache_rate: Dict[int, float] = {}
        process_branch_rate: Dict[int, float] = {}
        process_mem: Dict[int, int] = {}

        for pid in range(1, num_processes + 1):
            u = self.rng.random()
            mu = float(min_burst_us / (u ** (1.0 / alpha)))
            b_mean = max(min_burst_us, min(mu, 200000))
            process_burst_means[pid] = b_mean
            # Interactive processes (short burst) sleep longer; batch processes sleep shorter
            s_mean = max(1000.0, min(500000.0, 5_000_000.0 / b_mean))
            process_sleep_means[pid] = s_mean
            process_prios[pid] = 0 if b_mean < 2000 else (1 if b_mean < 10000 else 2)
            process_cache_rate[pid] = self.rng.uniform(0.01, 0.05)
            process_branch_rate[pid] = self.rng.uniform(0.005, 0.02)
            process_mem[pid] = int(self.rng.uniform(512, 16384))

        tasks: List[SimulatedTask] = []

        # Generate alternating burst-sleep sequences for each process
        for pid in range(1, num_processes + 1):
            t_curr = self.rng.randint(0, 10000)  # Staggered process start offsets
            for burst_idx in range(bursts_per_process):
                # Burst duration drawn from per-process lognormal distribution around mean
                b_factor = float(math.exp(self.rng.gauss(0.0, within_process_sigma)))
                burst = max(min_burst_us, int(process_burst_means[pid] * b_factor))

                # Sleep duration prior to this wakeup
                if burst_idx == 0:
                    sleep = 0
                else:
                    s_factor = float(math.exp(self.rng.gauss(0.0, 0.20)))
                    sleep = max(100, int(process_sleep_means[pid] * s_factor))

                tasks.append(
                    SimulatedTask(
                        pid=pid,
                        arrival_time_us=t_curr,
                        total_burst_us=burst,
                        sleep_time_us=sleep,
                        priority_level=process_prios[pid],
                        cache_miss_rate=process_cache_rate[pid],
                        branch_mispred_rate=process_branch_rate[pid],
                        memory_footprint_kb=process_mem[pid],
                    )
                )

                # Next burst arrives after execution plus sleep
                t_curr += burst + sleep

        # Sort all interleaved tasks by arrival time
        tasks.sort(key=lambda t: (t.arrival_time_us, t.pid))
        return tasks
