"""Realistic per-process burst prediction module for CPU scheduling without oracle knowledge."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ProcessBurstHistory:
    """Historical scheduling statistics tracked per PID across bursts."""

    pid: int
    ema_burst_us: float
    last_burst_us: float
    recent_bursts: List[float] = field(default_factory=list)
    last_completion_time_us: Optional[int] = None
    total_bursts_seen: int = 0


class BurstEstimator:
    """
    Realistic per-process CPU burst estimator.
    Combines:
    1. Exponential Moving Average (EMA) over prior bursts of the SAME PID
    2. Last-burst duration
    3. Rolling recent runtime average
    4. Wakeup context: sleep/blocked duration prior to wakeup, priority/nice level
    5. Monotonic residual runtime tracking during execution: max(100, estimate - elapsed)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        default_estimate_us: int = 5000,
        noise_std_frac: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.default_estimate_us = default_estimate_us
        self.noise_std_frac = max(0.0, noise_std_frac)
        self.rng = np.random.default_rng(seed)
        self._history: Dict[int, ProcessBurstHistory] = {}

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)

    def get_estimate(
        self,
        pid: int,
        elapsed_us: int = 0,
        sleep_time_us: int = 0,
        priority_level: int = 0,
    ) -> int:
        """
        Returns estimated residual burst length in microseconds.
        """
        if pid in self._history and self._history[pid].total_bursts_seen > 0:
            h = self._history[pid]
            base_estimate = h.ema_burst_us

            # Wakeup context: sleep time modulation (longer sleep -> interactive, shorter burst)
            if sleep_time_us > 0:
                sleep_factor = 1.0 - 0.20 * (sleep_time_us / (sleep_time_us + 20000.0))
                base_estimate *= sleep_factor

            # Priority modulation
            if priority_level != 0:
                base_estimate *= (1.0 + 0.05 * priority_level)
        else:
            # Unseen task prior adjusted by priority
            base_estimate = float(self.default_estimate_us)
            if priority_level != 0:
                base_estimate *= (1.0 + 0.10 * priority_level)

        if self.noise_std_frac > 0.0:
            noise = self.rng.normal(0.0, self.noise_std_frac * base_estimate)
            base_estimate = max(100.0, base_estimate + noise)

        if elapsed_us < base_estimate:
            residual = max(100.0, base_estimate - elapsed_us)
        else:
            # Heavy-tail / decreasing hazard rate: tasks that have executed past prior
            # are recognized as long batch jobs whose remaining duration scales with elapsed time
            residual = max(1000.0, elapsed_us * 0.5)

        return int(residual)

    def on_task_completion(
        self, pid: int, actual_burst_us: int, completion_time_us: Optional[int] = None
    ) -> None:
        """
        Update per-process moving statistics upon CPU burst completion.
        """
        actual_b = float(max(1, actual_burst_us))
        if pid not in self._history:
            prior = float(self.default_estimate_us)
            updated = self.alpha * actual_b + (1.0 - self.alpha) * prior
            self._history[pid] = ProcessBurstHistory(
                pid=pid,
                ema_burst_us=updated,
                last_burst_us=actual_b,
                recent_bursts=[actual_b],
                last_completion_time_us=completion_time_us,
                total_bursts_seen=1,
            )
        else:
            h = self._history[pid]
            h.ema_burst_us = self.alpha * actual_b + (1.0 - self.alpha) * h.ema_burst_us
            h.last_burst_us = actual_b
            h.recent_bursts.append(actual_b)
            if len(h.recent_bursts) > 5:
                h.recent_bursts.pop(0)
            h.last_completion_time_us = completion_time_us
            h.total_bursts_seen += 1

    def reset(self) -> None:
        self._history.clear()
