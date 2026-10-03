"""SMP NeuroOS-Lite learned multi-core scheduler (Phase 6).

Implements multi-core sched_ext dispatch logic:
- Per-CPU runqueues with cache-affinity placement.
- NUMA topology-aware candidate scoring and task affinity.
- Hierarchical work-stealing (intra-NUMA domain first, inter-NUMA fallback).
- Dynamic quantum sizing and fail-safe MLFQ guardrails.
- Integrated online drift self-correction.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
try:
    import numpy as np
except ImportError:
    np = None

from schedulers.base import BaseScheduler
from schedulers.mlfq.scheduler import MLFQScheduler
from simulator.scheduling.task import SimulatedTask
from userspace.trainer.burst_estimator import BurstEstimator
from userspace.trainer.observation import ObservationEncoder
from userspace.drift.online_corrector import OnlineDriftCorrector

try:
    from ml.quantization.lut import build_quantum_lut
    _LUT_AVAILABLE = True
except ImportError:
    _LUT_AVAILABLE = False


class SMPNeuroOSLiteScheduler:
    """Multi-core learned scheduler with NUMA awareness and work-stealing."""

    def __init__(
        self,
        num_cpus: int = 4,
        num_numa_nodes: int = 2,
        q_min_us: int = 1000,
        q_max_us: int = 50000,
        max_queue_depth: int = 1024,
        drift_sigma_threshold: float = 3.0,
        enable_online_drift_correction: bool = True,
        student_params: Optional[Dict[str, np.ndarray]] = None,
        quantized_policy: Optional[Dict[str, np.ndarray]] = None,
        quantized_scales: Optional[Dict[str, float]] = None,
    ) -> None:
        self.name = "SMP-NeuroOS-Lite"
        self.is_preemptive = True
        self.num_cpus = num_cpus
        self.num_numa_nodes = num_numa_nodes
        self.q_min_us = q_min_us
        self.q_max_us = q_max_us
        self.max_queue_depth = max_queue_depth
        self.drift_sigma_threshold = drift_sigma_threshold
        self.enable_online_drift_correction = enable_online_drift_correction

        self.cpu_to_node = [
            (c * self.num_numa_nodes) // self.num_cpus for c in range(self.num_cpus)
        ]

        self.runqueues: List[Deque[SimulatedTask]] = [deque() for _ in range(self.num_cpus)]
        self.fallbacks: List[MLFQScheduler] = [MLFQScheduler() for _ in range(self.num_cpus)]
        self.fallback_trips = 0
        self.steals_performed = 0

        self.burst_estimator = BurstEstimator()
        self.encoder = ObservationEncoder()
        self.online_corrector = OnlineDriftCorrector()
        self.student_params = student_params
        self.qpolicy = quantized_policy
        self.qscales = quantized_scales or {"s_x": 0.05, "s_w1": 0.05, "s_w2": 0.05, "s_b2": 0.001}

        if _LUT_AVAILABLE:
            self._lut = build_quantum_lut(q_min_us=q_min_us, q_max_us=q_max_us, num_bins=32)
        else:
            self._lut = None

    def configure_topology(self, num_cpus: int, num_numa_nodes: int, cpu_to_node: List[int]) -> None:
        self.num_cpus = num_cpus
        self.num_numa_nodes = num_numa_nodes
        self.cpu_to_node = list(cpu_to_node)
        while len(self.runqueues) < self.num_cpus:
            self.runqueues.append(deque())
            self.fallbacks.append(MLFQScheduler())

    def get_numa_distance(self, cpu_a: int, cpu_b: int) -> int:
        if cpu_a == cpu_b:
            return 10
        if self.cpu_to_node[cpu_a] == self.cpu_to_node[cpu_b]:
            return 12
        return 20

    def select_cpu(
        self,
        task: SimulatedTask,
        current_time_us: int,
        prev_cpu: int = 0,
        last_run_time_us: int = 0,
    ) -> int:
        """Select best CPU for incoming task balancing cache-warmth and queue depth."""
        if prev_cpu >= self.num_cpus:
            prev_cpu = 0

        is_cache_hot = (
            last_run_time_us > 0
            and (current_time_us - last_run_time_us) < 500
        )

        # If previous CPU is idle or light, keep cache warm
        if is_cache_hot and len(self.runqueues[prev_cpu]) == 0:
            return prev_cpu

        best_cpu = prev_cpu
        lowest_load = 1e9

        for c in range(self.num_cpus):
            q_len = len(self.runqueues[c])
            load = q_len * 10

            dist = self.get_numa_distance(prev_cpu, c)
            if dist > 12:
                load += 15  # Remote NUMA penalty
            elif dist > 10:
                load += 5   # Cross-core intra-node penalty

            if c == prev_cpu and is_cache_hot:
                load -= 8   # Cache warmth discount

            if load < lowest_load:
                lowest_load = load
                best_cpu = c

        return best_cpu

    def add_task(
        self,
        task: SimulatedTask,
        current_time_us: int,
        prev_cpu: int = 0,
        last_run_time_us: int = 0,
    ) -> None:
        target_cpu = self.select_cpu(task, current_time_us, prev_cpu, last_run_time_us)
        self.runqueues[target_cpu].append(task)
        self.fallbacks[target_cpu].add_task(task, current_time_us)

    def on_task_preempted(self, task: SimulatedTask, current_time_us: int, cpu_id: int = 0) -> None:
        self.runqueues[cpu_id].append(task)
        self.fallbacks[cpu_id].on_task_preempted(task, current_time_us)

    def on_task_completion(self, task: SimulatedTask, current_time_us: int, cpu_id: int = 0) -> None:
        predicted = self.burst_estimator.get_estimate(task.pid)
        self.burst_estimator.on_task_completion(
            task.pid, task.total_burst_us, completion_time_us=current_time_us
        )
        self.fallbacks[cpu_id].on_task_completion(task, current_time_us)

        # Feed actual burst to online drift corrector
        if self.enable_online_drift_correction:
            self.online_corrector.update(predicted, task.total_burst_us)

    def has_runnable_tasks(self) -> bool:
        return any(len(q) > 0 for q in self.runqueues)

    def pick_next_task_on_cpu(self, cpu_id: int, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        rq = self.runqueues[cpu_id]
        if not rq:
            return None, 0

        depth = len(rq)
        # Guardrail A: queue saturation
        if depth > self.max_queue_depth:
            self.fallback_trips += 1
            task = rq.popleft()
            fb_task, fb_q = self.fallbacks[cpu_id].pick_next_task(current_time_us)
            q = min(fb_q, task.remaining_burst_us) if fb_task is not None else self.q_min_us
            return task, max(1, int(q))

        # Guardrail B: drift detection
        if self.enable_online_drift_correction:
            drift_sigma = self.online_corrector.get_drift_sigma()
        else:
            drift_sigma = 0.0

        if drift_sigma > self.drift_sigma_threshold:
            self.fallback_trips += 1
            task = rq.popleft()
            fb_task, fb_q = self.fallbacks[cpu_id].pick_next_task(current_time_us)
            q = min(fb_q, task.remaining_burst_us) if fb_task is not None else self.q_min_us
            return task, max(1, int(q))

        # Neural scoring
        cands = list(rq)[: self.encoder.top_k]
        if not cands:
            return None, 0

        mat, mask = self.encoder.encode(
            ready_tasks=cands,
            running_task=None,
            current_time_us=current_time_us,
            burst_estimator=self.burst_estimator,
        )
        n = int(mask.sum())
        mat = mat[:n]

        if self.qpolicy is not None and np is not None:
            from ml.quantization.quantize import quantized_forward_int
            meta = {"scales": self.qscales}
            scores = list(quantized_forward_int(self.qpolicy, meta, mat.astype(np.float64)))
        elif self.student_params is not None and np is not None:
            from ml.distillation.distiller import numpy_student_forward
            scores = list(numpy_student_forward(self.student_params, mat.astype(np.float64)))
        else:
            scores = [-float(row[1]) + 0.5 * float(row[2]) for row in mat]

        # Apply online residual drift correction to score
        if self.enable_online_drift_correction:
            scores = [self.online_corrector.adjust_score(float(s)) for s in scores]

        idx = max(range(len(scores)), key=lambda i: scores[i])
        task = cands[idx]
        try:
            rq.remove(task)
        except ValueError:
            pass

        # Quantum calculation
        if self._lut is not None:
            quantum = int(self._lut.score_to_quantum(float(scores[idx])))
        else:
            quantum = int(self.q_min_us + 0.5 * (self.q_max_us - self.q_min_us))

        quantum = max(self.q_min_us, min(self.q_max_us, quantum))
        quantum = min(quantum, task.remaining_burst_us)
        return task, max(1, quantum)


    def steal_work_for_cpu(self, idle_cpu_id: int, current_time_us: int) -> Tuple[Optional[SimulatedTask], int]:
        """Hierarchical work stealing: intra-NUMA first, then inter-NUMA."""
        idle_node = self.cpu_to_node[idle_cpu_id]

        best_victim = -1
        max_depth = 0

        # Step 1: Intra-NUMA search
        for c in range(self.num_cpus):
            if c == idle_cpu_id:
                continue
            if self.cpu_to_node[c] == idle_node:
                depth = len(self.runqueues[c])
                if depth >= 2 and depth > max_depth:
                    max_depth = depth
                    best_victim = c

        # Step 2: Inter-NUMA search if intra-NUMA found no viable victim
        if best_victim == -1:
            for c in range(self.num_cpus):
                if c == idle_cpu_id:
                    continue
                if self.cpu_to_node[c] != idle_node:
                    depth = len(self.runqueues[c])
                    if depth >= 3 and depth > max_depth:
                        max_depth = depth
                        best_victim = c

        if best_victim >= 0 and self.runqueues[best_victim]:
            # Steal from tail of victim queue (avoids stealing currently hot task)
            stolen_task = self.runqueues[best_victim].pop()
            self.steals_performed += 1
            quantum = min(self.q_min_us * 2, stolen_task.remaining_burst_us)
            return stolen_task, max(1, quantum)

        return None, 0
