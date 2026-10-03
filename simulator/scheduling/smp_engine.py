"""Multi-Core SMP and NUMA-aware scheduling simulation engine (Phase 6).

Extends the uniprocessor cycle-accurate simulation engine to symmetric multiprocessing (SMP):
- M CPU cores grouped across K NUMA nodes.
- Per-core runqueues and independent dispatch loops.
- Cache-warmth tracking and migration penalties (intra-NUMA vs cross-NUMA).
- Work-stealing load balancer with thresholding.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
try:
    import numpy as np
except ImportError:
    np = None

from simulator.scheduling.task import SimulatedTask, TaskState


@dataclass
class SMPSimulationMetrics:
    """Summary metrics produced by a multi-core SMP simulation run."""

    num_cpus: int
    num_numa_nodes: int
    total_tasks: int
    completed_tasks: int
    total_makespan_us: int
    per_cpu_busy_time_us: List[int]
    overall_cpu_utilization: float

    mean_turnaround_time_us: float
    mean_normalized_turnaround_time: float
    mean_waiting_time_us: float
    mean_response_time_us: float

    p95_waiting_time_us: float
    p99_waiting_time_us: float
    p999_waiting_time_us: float

    total_context_switches: int
    total_migrations: int
    intra_node_migrations: int
    inter_node_migrations: int
    migration_overhead_us: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "num_cpus": float(self.num_cpus),
            "num_numa_nodes": float(self.num_numa_nodes),
            "total_tasks": float(self.total_tasks),
            "completed_tasks": float(self.completed_tasks),
            "total_makespan_us": float(self.total_makespan_us),
            "overall_cpu_utilization": self.overall_cpu_utilization,
            "mean_turnaround_time_us": self.mean_turnaround_time_us,
            "mean_normalized_turnaround_time": self.mean_normalized_turnaround_time,
            "mean_waiting_time_us": self.mean_waiting_time_us,
            "mean_response_time_us": self.mean_response_time_us,
            "p95_waiting_time_us": self.p95_waiting_time_us,
            "p99_waiting_time_us": self.p99_waiting_time_us,
            "p999_waiting_time_us": self.p999_waiting_time_us,
            "total_context_switches": float(self.total_context_switches),
            "total_migrations": float(self.total_migrations),
            "intra_node_migrations": float(self.intra_node_migrations),
            "inter_node_migrations": float(self.inter_node_migrations),
            "migration_overhead_us": float(self.migration_overhead_us),
        }


class SMPSchedulingSimulationEngine:
    """
    Multi-core discrete-event simulation engine with NUMA topology.
    Simulates SMP execution, per-core time-slicing, work stealing, and migration overhead.
    """

    def __init__(
        self,
        scheduler,
        num_cpus: int = 4,
        num_numa_nodes: int = 2,
        context_switch_overhead_us: int = 2,
        intra_node_migration_us: int = 5,
        inter_node_migration_us: int = 15,
        cache_hot_threshold_us: int = 500,
    ) -> None:
        self.scheduler = scheduler
        self.num_cpus = max(1, num_cpus)
        self.num_numa_nodes = max(1, min(num_numa_nodes, self.num_cpus))
        self.context_switch_overhead_us = max(0, context_switch_overhead_us)
        self.intra_node_migration_us = intra_node_migration_us
        self.inter_node_migration_us = inter_node_migration_us
        self.cache_hot_threshold_us = cache_hot_threshold_us

        # Build CPU -> NUMA node mapping
        self.cpu_to_node = [
            (c * self.num_numa_nodes) // self.num_cpus for c in range(self.num_cpus)
        ]

    def get_numa_distance(self, cpu_a: int, cpu_b: int) -> int:
        if cpu_a == cpu_b:
            return 10
        if self.cpu_to_node[cpu_a] == self.cpu_to_node[cpu_b]:
            return 12  # Same NUMA node, different core
        return 20      # Cross-NUMA node

    def run(self, workload: List[SimulatedTask]) -> Tuple[List[SimulatedTask], SMPSimulationMetrics]:
        pending_arrivals = sorted(
            [copy.deepcopy(t) for t in workload], key=lambda t: (t.arrival_time_us, t.pid)
        )

        completed_tasks: List[SimulatedTask] = []
        current_time_us = 0

        running_tasks: List[Optional[SimulatedTask]] = [None] * self.num_cpus
        slice_remaining: List[int] = [0] * self.num_cpus
        last_pid_on_cpu: List[Optional[int]] = [None] * self.num_cpus
        last_run_time_of_pid: Dict[int, int] = {}
        last_cpu_of_pid: Dict[int, int] = {}

        per_cpu_busy_time: List[int] = [0] * self.num_cpus
        total_context_switches = 0
        total_migrations = 0
        intra_node_migrations = 0
        inter_node_migrations = 0
        total_migration_overhead_us = 0

        # Inform scheduler of topology if supported
        if hasattr(self.scheduler, "configure_topology"):
            self.scheduler.configure_topology(self.num_cpus, self.num_numa_nodes, self.cpu_to_node)

        while pending_arrivals or self.scheduler.has_runnable_tasks() or any(t is not None for t in running_tasks):
            # 1. Admit newly arrived tasks and assign to CPUs
            while pending_arrivals and pending_arrivals[0].arrival_time_us <= current_time_us:
                arrived = pending_arrivals.pop(0)
                arrived.state = TaskState.READY
                prev_cpu = last_cpu_of_pid.get(arrived.pid, 0)
                last_time = last_run_time_of_pid.get(arrived.pid, 0)
                self.scheduler.add_task(arrived, current_time_us, prev_cpu=prev_cpu, last_run_time_us=last_time)

            # 2. Per-core dispatch for any idle core
            for c in range(self.num_cpus):
                if running_tasks[c] is None:
                    next_task, quantum_us = self.scheduler.pick_next_task_on_cpu(c, current_time_us)

                    # Work stealing if core queue was empty
                    if next_task is None and hasattr(self.scheduler, "steal_work_for_cpu"):
                        next_task, quantum_us = self.scheduler.steal_work_for_cpu(c, current_time_us)

                    if next_task is not None:
                        # Check migration from another CPU
                        if next_task.pid in last_cpu_of_pid:
                            p_cpu = last_cpu_of_pid[next_task.pid]
                            if p_cpu != c:
                                total_migrations += 1
                                is_same_node = (self.cpu_to_node[p_cpu] == self.cpu_to_node[c])
                                mig_cost = self.intra_node_migration_us if is_same_node else self.inter_node_migration_us
                                is_hot = (current_time_us - last_run_time_of_pid.get(next_task.pid, 0)) < self.cache_hot_threshold_us
                                if is_hot:
                                    mig_cost *= 2
                                if is_same_node:
                                    intra_node_migrations += 1
                                else:
                                    inter_node_migrations += 1
                                total_migration_overhead_us += mig_cost

                        # Context switch penalty
                        if last_pid_on_cpu[c] is not None and last_pid_on_cpu[c] != next_task.pid:
                            total_context_switches += 1
                            next_task.context_switches += 1

                        running_tasks[c] = next_task
                        running_tasks[c].state = TaskState.RUNNING
                        slice_remaining[c] = quantum_us
                        last_pid_on_cpu[c] = next_task.pid
                        last_cpu_of_pid[next_task.pid] = c
                        last_run_time_of_pid[next_task.pid] = current_time_us

                        if next_task.first_dispatch_time_us is None:
                            next_task.first_dispatch_time_us = current_time_us

            # 3. Determine next event step
            active_cores = [c for c in range(self.num_cpus) if running_tasks[c] is not None]
            if not active_cores:
                if pending_arrivals:
                    current_time_us = pending_arrivals[0].arrival_time_us
                    continue
                else:
                    break

            # Find min step among all executing tasks
            step_candidates = []
            for c in active_cores:
                t = running_tasks[c]
                assert t is not None
                rem_burst = t.remaining_burst_us
                rem_slice = slice_remaining[c] if slice_remaining[c] > 0 else rem_burst
                step_candidates.append(min(rem_burst, rem_slice))

            if pending_arrivals:
                time_to_arrival = max(1, pending_arrivals[0].arrival_time_us - current_time_us)
                step_candidates.append(time_to_arrival)

            step = max(1, min(step_candidates))

            # 4. Advance execution across all active cores
            current_time_us += step
            for c in active_cores:
                t = running_tasks[c]
                assert t is not None
                per_cpu_busy_time[c] += step
                t.executed_burst_us += step
                t.remaining_burst_us -= step
                slice_remaining[c] -= step
                last_run_time_of_pid[t.pid] = current_time_us

                # Check task completion
                if t.remaining_burst_us <= 0:
                    t.state = TaskState.COMPLETED
                    t.completion_time_us = current_time_us
                    tt = t.turnaround_time_us
                    assert tt is not None
                    t.waiting_time_us = tt - t.total_burst_us
                    self.scheduler.on_task_completion(t, current_time_us, cpu_id=c)
                    completed_tasks.append(t)
                    running_tasks[c] = None
                    slice_remaining[c] = 0

                # Check time slice expiration
                elif slice_remaining[c] <= 0:
                    t.state = TaskState.READY
                    self.scheduler.on_task_preempted(t, current_time_us, cpu_id=c)
                    running_tasks[c] = None
                    slice_remaining[c] = 0

        # Calculate metrics
        metrics = self._compute_metrics(
            completed_tasks=completed_tasks,
            total_makespan_us=max(1, current_time_us),
            per_cpu_busy_time_us=per_cpu_busy_time,
            total_context_switches=total_context_switches,
            total_migrations=total_migrations,
            intra_node_migrations=intra_node_migrations,
            inter_node_migrations=inter_node_migrations,
            migration_overhead_us=total_migration_overhead_us,
        )

        return completed_tasks, metrics

    def _compute_metrics(
        self,
        completed_tasks: List[SimulatedTask],
        total_makespan_us: int,
        per_cpu_busy_time_us: List[int],
        total_context_switches: int,
        total_migrations: int,
        intra_node_migrations: int,
        inter_node_migrations: int,
        migration_overhead_us: int,
    ) -> SMPSimulationMetrics:
        n = len(completed_tasks)
        total_busy = sum(per_cpu_busy_time_us)
        total_capacity = max(1, total_makespan_us * self.num_cpus)
        utilization = float(total_busy) / float(total_capacity)

        if n == 0:
            return SMPSimulationMetrics(
                num_cpus=self.num_cpus,
                num_numa_nodes=self.num_numa_nodes,
                total_tasks=0,
                completed_tasks=0,
                total_makespan_us=total_makespan_us,
                per_cpu_busy_time_us=per_cpu_busy_time_us,
                overall_cpu_utilization=0.0,
                mean_turnaround_time_us=0.0,
                mean_normalized_turnaround_time=0.0,
                mean_waiting_time_us=0.0,
                mean_response_time_us=0.0,
                p95_waiting_time_us=0.0,
                p99_waiting_time_us=0.0,
                p999_waiting_time_us=0.0,
                total_context_switches=total_context_switches,
                total_migrations=total_migrations,
                intra_node_migrations=intra_node_migrations,
                inter_node_migrations=inter_node_migrations,
                migration_overhead_us=migration_overhead_us,
            )

        tats = [t.turnaround_time_us for t in completed_tasks if t.turnaround_time_us is not None]
        ntats = [t.normalized_turnaround_time for t in completed_tasks if t.normalized_turnaround_time is not None]
        waits = [float(t.waiting_time_us) for t in completed_tasks]
        resps = [float(t.response_time_us) for t in completed_tasks if t.response_time_us is not None]

        def _pct(arr: List[float], q: float) -> float:
            if not arr:
                return 0.0
            if np is not None:
                return float(np.percentile(arr, q))
            s = sorted(arr)
            idx = int((q / 100.0) * (len(s) - 1))
            return float(s[idx])

        def _avg(arr) -> float:
            if not arr:
                return 0.0
            if np is not None:
                return float(np.mean(arr))
            return float(sum(arr)) / float(len(arr))

        p95 = _pct(waits, 95)
        p99 = _pct(waits, 99)
        p999 = _pct(waits, 99.9)

        return SMPSimulationMetrics(
            num_cpus=self.num_cpus,
            num_numa_nodes=self.num_numa_nodes,
            total_tasks=n,
            completed_tasks=n,
            total_makespan_us=total_makespan_us,
            per_cpu_busy_time_us=per_cpu_busy_time_us,
            overall_cpu_utilization=utilization,
            mean_turnaround_time_us=_avg(tats),
            mean_normalized_turnaround_time=_avg(ntats),
            mean_waiting_time_us=_avg(waits),
            mean_response_time_us=_avg(resps),
            p95_waiting_time_us=p95,
            p99_waiting_time_us=p99,
            p999_waiting_time_us=p999,
            total_context_switches=total_context_switches,
            total_migrations=total_migrations,
            intra_node_migrations=intra_node_migrations,
            inter_node_migrations=inter_node_migrations,
            migration_overhead_us=migration_overhead_us,
        )

