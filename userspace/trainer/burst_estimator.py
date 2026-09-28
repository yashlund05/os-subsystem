"""Burst estimation module for CPU scheduling without oracle knowledge."""

from typing import Dict, Optional

import numpy as np


class BurstEstimator:
    """
    Exponential-smoothing burst estimator with optional noise injection.
    Models runtime kernel burst prediction without accessing oracle future burst lengths.

    Formula:
        tau_{n+1} = alpha * t_n + (1 - alpha) * tau_n
    """

    def __init__(
        self,
        alpha: float = 0.5,
        default_estimate_us: int = 5000,
        noise_std_frac: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.default_estimate_us = default_estimate_us
        self.noise_std_frac = max(0.0, noise_std_frac)
        self.rng = np.random.default_rng(seed)
        self._history: Dict[int, float] = {}

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)

    def get_estimate(self, pid: int, elapsed_us: int = 0) -> int:
        """
        Returns estimated residual burst length in microseconds.
        """
        base_estimate = self._history.get(pid, float(self.default_estimate_us))

        if self.noise_std_frac > 0.0:
            noise = self.rng.normal(0.0, self.noise_std_frac * base_estimate)
            base_estimate = max(100.0, base_estimate + noise)

        # Residual estimate cannot be less than 0
        residual = max(100.0, base_estimate - elapsed_us)
        return int(residual)

    def on_task_completion(self, pid: int, actual_burst_us: int) -> None:
        """
        Update exponential moving average upon burst completion.
        """
        prior = self._history.get(pid, float(self.default_estimate_us))
        updated = self.alpha * actual_burst_us + (1.0 - self.alpha) * prior
        self._history[pid] = updated

    def reset(self) -> None:
        self._history.clear()
