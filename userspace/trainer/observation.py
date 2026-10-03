"""Observation encoder mapping ready-queue tasks to kernel-compatible 16-D feature vectors."""

from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    class _FallbackArray(list):
        @property
        def shape(self):
            return (len(self), len(self[0])) if self and isinstance(self[0], list) else (len(self),)
        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                row, col_slice = key
                if isinstance(col_slice, slice):
                    start = col_slice.start or 0
                    stop = col_slice.stop if col_slice.stop is not None else len(self[row])
                    self[row][start:stop] = value
                    return
            super().__setitem__(key, value)
        def sum(self):
            return sum(self)
        def astype(self, dtype):
            return self

    class _NumpyFallback:
        float32 = float
        float64 = float
        int8 = int
        int32 = int
        ndarray = _FallbackArray
        @staticmethod
        def zeros(shape, dtype=float):
            if isinstance(shape, tuple) and len(shape) > 1:
                return _FallbackArray([_FallbackArray([0.0] * shape[1]) for _ in range(shape[0])])
            size = shape[0] if isinstance(shape, tuple) else shape
            return _FallbackArray([0.0] * size)

        @staticmethod
        def clip(val, lo, hi):
            return max(lo, min(hi, val))
        @staticmethod
        def mean(arr):
            return float(sum(arr)) / float(len(arr)) if arr else 0.0
        @staticmethod
        def percentile(arr, q):
            if not arr: return 0.0
            s = sorted(arr)
            return float(s[int((q / 100.0) * (len(s) - 1))])
        @staticmethod
        def argmax(arr):
            return max(range(len(arr)), key=lambda i: arr[i])
        @staticmethod
        def array(arr):
            return _FallbackArray(arr)

    np = _NumpyFallback()

from simulator.scheduling.task import SimulatedTask
from userspace.trainer.burst_estimator import BurstEstimator

# Exact match with NEUROOS_INPUT_DIM in kernel/include/neuroos_kernel.h
NEUROOS_INPUT_DIM = 16
DEFAULT_TOP_K = 16


class ObservationEncoder:
    """
    Encodes ready-queue candidates into fixed-size padded matrices.
    Each candidate is represented by exactly NEUROOS_INPUT_DIM (16) normalized features:
      10 task-specific features + 6 global context features.
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        max_burst_us: float = 100000.0,
        max_wait_us: float = 500000.0,
        max_pmu_delta: float = 5000.0,
        max_mem_kb: float = 65536.0,
        max_queue_depth: float = 1024.0,
    ) -> None:
        self.top_k = top_k
        self.max_burst_us = max_burst_us
        self.max_wait_us = max_wait_us
        self.max_pmu_delta = max_pmu_delta
        self.max_mem_kb = max_mem_kb
        self.max_queue_depth = max_queue_depth

    def encode(
        self,
        ready_tasks: List[SimulatedTask],
        running_task: Optional[SimulatedTask],
        current_time_us: int,
        burst_estimator: BurstEstimator,
        load_factor: float = 0.8,
        total_context_switches: int = 0,
        last_switch_time_us: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes up to top_k candidates into:
        - candidates: shape (top_k, 16) float32
        - action_mask: shape (top_k,) int8 (1 for valid task, 0 for empty slot)
        """
        candidates = np.zeros((self.top_k, NEUROOS_INPUT_DIM), dtype=np.float32)
        action_mask = np.zeros((self.top_k,), dtype=np.int8)

        # Candidate list: active running task (if any) first, then ready queue tasks
        all_candidates: List[Tuple[SimulatedTask, bool]] = []
        if running_task is not None:
            all_candidates.append((running_task, True))

        for t in ready_tasks:
            all_candidates.append((t, False))

        num_valid = min(self.top_k, len(all_candidates))

        # Global features (6 dimensions)
        queue_len = len(ready_tasks) + (1 if running_task else 0)
        queue_len_norm = float(np.clip(queue_len / self.max_queue_depth, 0.0, 5.0))
        load_factor_norm = float(np.clip(load_factor, 0.0, 2.0))
        cpu_busy_frac = 1.0 if running_task is not None else 0.0

        max_wait = 0.0
        sum_pred_burst = 0.0
        running_pred_rem = 0.0
        if running_task is not None:
            r_pred = burst_estimator.get_estimate(
                running_task.pid,
                running_task.executed_burst_us,
                running_task.sleep_time_us,
                running_task.priority_level,
            )
            running_pred_rem = max(100.0, float(r_pred - running_task.executed_burst_us))

        for task, _is_running in all_candidates:
            wait_time = max(0, current_time_us - task.arrival_time_us - task.executed_burst_us)
            if wait_time > max_wait:
                max_wait = float(wait_time)
            pred_b = burst_estimator.get_estimate(
                task.pid, task.executed_burst_us, task.sleep_time_us, task.priority_level
            )
            sum_pred_burst += pred_b

        max_wait_norm = float(np.clip(max_wait / self.max_wait_us, 0.0, 5.0))
        mean_pred_norm = float(
            np.clip((sum_pred_burst / max(1, len(all_candidates))) / self.max_burst_us, 0.0, 5.0)
        )
        running_rem_norm = float(np.clip(running_pred_rem / self.max_burst_us, 0.0, 5.0))

        global_features = [
            queue_len_norm,
            load_factor_norm,
            cpu_busy_frac,
            max_wait_norm,
            mean_pred_norm,
            running_rem_norm,
        ]

        # Populate candidate slots
        for idx in range(num_valid):
            task, is_running = all_candidates[idx]
            action_mask[idx] = 1

            # Task features (10 dimensions)
            elapsed_norm = float(np.clip(task.executed_burst_us / self.max_burst_us, 0.0, 5.0))
            pred_burst = burst_estimator.get_estimate(
                task.pid, task.executed_burst_us, task.sleep_time_us, task.priority_level
            )
            pred_burst_norm = float(np.clip(pred_burst / self.max_burst_us, 0.0, 5.0))
            age_us = max(0, current_time_us - task.arrival_time_us - task.executed_burst_us)
            age_norm = float(np.clip(age_us / self.max_wait_us, 0.0, 5.0))
            ctx_switches_norm = float(np.clip(task.context_switches / 50.0, 0.0, 5.0))

            cache_miss_norm = float(np.clip(task.cache_misses / self.max_pmu_delta, 0.0, 5.0))
            branch_mispred_norm = float(
                np.clip(task.branch_mispredictions / self.max_pmu_delta, 0.0, 5.0)
            )
            mem_kb_norm = float(np.clip(task.memory_footprint_kb / self.max_mem_kb, 0.0, 5.0))
            priority_norm = float(np.clip(task.priority_level / 10.0, 0.0, 5.0))
            is_running_val = 1.0 if is_running else 0.0

            # Bounded, monotonic remaining-time estimate: clip((pred_burst - elapsed) / max_burst, 0.0, 1.0)
            burst_ratio = float(
                np.clip((pred_burst - task.executed_burst_us) / self.max_burst_us, 0.0, 1.0)
            )

            task_features = [
                elapsed_norm,
                pred_burst_norm,
                age_norm,
                ctx_switches_norm,
                cache_miss_norm,
                branch_mispred_norm,
                mem_kb_norm,
                priority_norm,
                is_running_val,
                burst_ratio,
            ]

            # Combined exact 16-D feature vector per candidate
            candidates[idx, :10] = task_features
            candidates[idx, 10:] = global_features

        return candidates, action_mask
