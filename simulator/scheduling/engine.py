"""Discrete-event uniprocessor CPU scheduling simulation engine."""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from schedulers.base import BaseScheduler
from simulator.scheduling.task import SimulatedTask, TaskState


@dataclass
class SimulationMetrics:
    """Summary metrics produced by a simulation run."""

    total_tasks: int
    completed_tasks: int
    total_makespan_us: int
    total_cpu_busy_time_us: int
    cpu_utilization: float

    mean_turnaround_time_us: float
    mean_normalized_turnaround_time: float
    mean_waiting_time_us: float
    mean_response_time_us: float

    p95_waiting_time_us: float
    p99_waiting_time_us: float
    p999_waiting_time_us: float

    total_context_switches: int
    context_switch_overhead_us: int
    overhead_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_tasks": float(self.total_tasks),
            "completed_tasks": float(self.completed_tasks),
            "total_makespan_us": float(self.total_makespan_us),
            "cpu_utilization": self.cpu_utilization,
            "mean_turnaround_time_us": self.mean_turnaround_time_us,
            "mean_normalized_turnaround_time": self.mean_normalized_turnaround_time,
            "mean_waiting_time_us": self.mean_waiting_time_us,
            "mean_response_time_us": self.mean_response_time_us,
            "p95_waiting_time_us": self.p95_waiting_time_us,
            "p99_waiting_time_us": self.p99_waiting_time_us,
            "p999_waiting_time_us": self.p999_waiting_time_us,
            "total_context_switches": float(self.total_context_switches),
            "overhead_ratio": self.overhead_ratio,
        }


class SchedulingSimulationEngine:
    """
    Cycle-accurate discrete-event scheduling simulation engine.
    Simulates uniprocessor scheduling execution, preemption, context-switch penalties,
    and telemetry progression.
    """

    def __init__(self, scheduler: BaseScheduler, context_switch_overhead_us: int = 2) -> None:
        self.scheduler = scheduler
        self.context_switch_overhead_us = max(0, context_switch_overhead_us)

    def run(self, workload: List[SimulatedTask]) -> Tuple[List[SimulatedTask], SimulationMetrics]:
        """
        Execute discrete-event simulation for the supplied workload.
        Returns:
            Tuple of (completed_tasks, metrics_summary)
        """
        # Deepcopy tasks so original workload is untouched
        pending_arrivals = sorted(
            [copy.deepcopy(t) for t in workload], key=lambda t: (t.arrival_time_us, t.pid)
        )

        completed_tasks: List[SimulatedTask] = []
        current_time_us = 0
        total_context_switches = 0
        total_cpu_busy_time_us = 0
        last_running_pid: Optional[int] = None

        running_task: Optional[SimulatedTask] = None
        current_slice_remaining_us = 0

        while pending_arrivals or self.scheduler.has_runnable_tasks() or running_task is not None:
            # 1. Admit all tasks that have arrived by current_time_us
            while pending_arrivals and pending_arrivals[0].arrival_time_us <= current_time_us:
                arrived_task = pending_arrivals.pop(0)
                arrived_task.state = TaskState.READY

                # Preemption check for SRTF or priority schedulers
                if running_task is not None and self.scheduler.is_preemptive:
                    if hasattr(self.scheduler, "should_preempt") and self.scheduler.should_preempt(
                        running_task, arrived_task
                    ):
                        # Preempt currently executing task
                        running_task.state = TaskState.READY
                        self.scheduler.on_task_preempted(running_task, current_time_us)
                        running_task = None
                        current_slice_remaining_us = 0

                self.scheduler.add_task(arrived_task, current_time_us)

            # 2. If CPU is idle, dispatch next task
            if running_task is None:
                if self.scheduler.has_runnable_tasks():
                    next_task, quantum_us = self.scheduler.pick_next_task(current_time_us)
                    if next_task is not None:
                        # Context switch penalty if switching to a different task
                        if last_running_pid is not None and last_running_pid != next_task.pid:
                            current_time_us += self.context_switch_overhead_us
                            total_context_switches += 1
                            next_task.context_switches += 1

                        running_task = next_task
                        running_task.state = TaskState.RUNNING
                        last_running_pid = running_task.pid
                        current_slice_remaining_us = quantum_us

                        if running_task.first_dispatch_time_us is None:
                            running_task.first_dispatch_time_us = current_time_us
                else:
                    # CPU is completely idle, fast-forward to next arrival
                    if pending_arrivals:
                        current_time_us = pending_arrivals[0].arrival_time_us
                        continue
                    else:
                        break

            # 3. Determine time step until next event
            # Next event is the minimum of:
            # - Task burst completion (running_task.remaining_burst_us)
            # - Quantum expiration (current_slice_remaining_us)
            # - Next pending task arrival
            time_to_completion = running_task.remaining_burst_us
            time_to_quantum_end = (
                current_slice_remaining_us if current_slice_remaining_us > 0 else time_to_completion
            )

            step = min(time_to_completion, time_to_quantum_end)

            if pending_arrivals:
                time_to_arrival = max(0, pending_arrivals[0].arrival_time_us - current_time_us)
                if time_to_arrival > 0:
                    step = min(step, time_to_arrival)

            step = max(1, step)

            # 4. Advance execution by step
            current_time_us += step
            total_cpu_busy_time_us += step
            running_task.executed_burst_us += step
            running_task.remaining_burst_us -= step
            current_slice_remaining_us -= step

            self.scheduler.on_tick(current_time_us, step)

            # 5. Check if task completed
            if running_task.remaining_burst_us <= 0:
                running_task.state = TaskState.COMPLETED
                running_task.completion_time_us = current_time_us
                running_task.waiting_time_us = (
                    running_task.turnaround_time_us - running_task.total_burst_us
                )
                self.scheduler.on_task_completion(running_task, current_time_us)
                completed_tasks.append(running_task)
                running_task = None
                current_slice_remaining_us = 0

            # 6. Check if quantum expired
            elif current_slice_remaining_us <= 0 and self.scheduler.is_preemptive:
                running_task.state = TaskState.READY
                self.scheduler.on_task_preempted(running_task, current_time_us)
                running_task = None
                current_slice_remaining_us = 0

        # Calculate summary metrics per docs/Metrics.md
        metrics = self._compute_metrics(
            completed_tasks=completed_tasks,
            total_makespan_us=max(1, current_time_us),
            total_cpu_busy_time_us=total_cpu_busy_time_us,
            total_context_switches=total_context_switches,
        )

        return completed_tasks, metrics

    def _compute_metrics(
        self,
        completed_tasks: List[SimulatedTask],
        total_makespan_us: int,
        total_cpu_busy_time_us: int,
        total_context_switches: int,
    ) -> SimulationMetrics:
        n = len(completed_tasks)
        if n == 0:
            return SimulationMetrics(
                total_tasks=0,
                completed_tasks=0,
                total_makespan_us=total_makespan_us,
                total_cpu_busy_time_us=0,
                cpu_utilization=0.0,
                mean_turnaround_time_us=0.0,
                mean_normalized_turnaround_time=0.0,
                mean_waiting_time_us=0.0,
                mean_response_time_us=0.0,
                p95_waiting_time_us=0.0,
                p99_waiting_time_us=0.0,
                p999_waiting_time_us=0.0,
                total_context_switches=total_context_switches,
                context_switch_overhead_us=self.context_switch_overhead_us,
                overhead_ratio=0.0,
            )

        tats = [t.turnaround_time_us for t in completed_tasks if t.turnaround_time_us is not None]
        ntats = [
            t.normalized_turnaround_time
            for t in completed_tasks
            if t.normalized_turnaround_time is not None
        ]
        wts = [max(0, t.waiting_time_us) for t in completed_tasks]
        rts = [t.response_time_us for t in completed_tasks if t.response_time_us is not None]

        mean_tat = float(np.mean(tats)) if tats else 0.0
        mean_ntat = float(np.mean(ntats)) if ntats else 0.0
        mean_wt = float(np.mean(wts)) if wts else 0.0
        mean_rt = float(np.mean(rts)) if rts else 0.0

        p95_wt = float(np.percentile(wts, 95)) if wts else 0.0
        p99_wt = float(np.percentile(wts, 99)) if wts else 0.0
        p999_wt = float(np.percentile(wts, 99.9)) if wts else 0.0

        cpu_util = min(1.0, total_cpu_busy_time_us / total_makespan_us)

        # Context switch overhead penalty ratio Phi_overhead
        total_ctx_overhead = total_context_switches * self.context_switch_overhead_us
        overhead_ratio = total_ctx_overhead / total_makespan_us

        return SimulationMetrics(
            total_tasks=n,
            completed_tasks=n,
            total_makespan_us=total_makespan_us,
            total_cpu_busy_time_us=total_cpu_busy_time_us,
            cpu_utilization=cpu_util,
            mean_turnaround_time_us=mean_tat,
            mean_normalized_turnaround_time=mean_ntat,
            mean_waiting_time_us=mean_wt,
            mean_response_time_us=mean_rt,
            p95_waiting_time_us=p95_wt,
            p99_waiting_time_us=p99_wt,
            p999_waiting_time_us=p999_wt,
            total_context_switches=total_context_switches,
            context_switch_overhead_us=self.context_switch_overhead_us,
            overhead_ratio=overhead_ratio,
        )
