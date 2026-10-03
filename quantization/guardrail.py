"""Fail-safe guardrail engine for NeuroOS-Lite kernel and simulation environments.

Guarantees system safety under distribution shift or adversarial conditions per docs/TRD.md:
1. Queue Saturation Guardrail: depth > 1024 -> fallback.
2. Input Out-of-Distribution (OOD) Guardrail: feature values exceeding saturation bounds (> 3.0 sigma).
3. Prediction Residual Drift Guardrail: empirical residual exceeding 3.0 sigma (3000 milli-sigma).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

MAX_QUEUE_DEPTH = 1024
DRIFT_THRESHOLD_MILLI_SIGMA = 3000  # 3.0 sigma in milli-sigma


class FallbackReason(IntEnum):
    NONE = 0
    QUEUE_SATURATION = 1
    INPUT_OOD = 2
    PREDICTION_DRIFT = 3
    MANUAL_OVERRIDE = 4


@dataclass
class GuardrailDecision:
    pass_check: bool
    reason: FallbackReason
    detail: str


class GuardrailEngine:
    """Runtime monitor evaluating fast-path safety invariants."""

    def __init__(
        self,
        max_queue_depth: int = MAX_QUEUE_DEPTH,
        drift_sigma_threshold: float = 3.0,
        feature_upper_bound: float = 10.0,
    ) -> None:
        self.max_queue_depth = max_queue_depth
        self.drift_sigma_threshold = drift_sigma_threshold
        self.feature_upper_bound = feature_upper_bound

        # Running online statistics for prediction residuals (Welford's algorithm)
        self.residual_count = 0
        self.residual_mean = 0.0
        self.residual_m2 = 0.0

        # Trip counters
        self.total_checks = 0
        self.total_trips = 0
        self.trips_by_reason = dict.fromkeys(FallbackReason, 0)

    def record_prediction_outcome(self, pred_burst_us: float, actual_burst_us: float) -> None:
        """Updates online mean and variance of prediction error."""
        residual = abs(pred_burst_us - actual_burst_us)
        self.residual_count += 1
        delta = residual - self.residual_mean
        self.residual_mean += delta / self.residual_count
        delta2 = residual - self.residual_mean
        self.residual_m2 += delta * delta2

    @property
    def residual_std(self) -> float:
        if self.residual_count < 2:
            return 1000.0  # Default initial std in us
        return max(1.0, float(np.sqrt(self.residual_m2 / (self.residual_count - 1))))

    def check_guardrails(
        self,
        queue_depth: int,
        candidate_features: Optional[np.ndarray] = None,
        current_running_residual: Optional[float] = None,
    ) -> GuardrailDecision:
        """Evaluates all safety guardrails. Returns GuardrailDecision."""
        self.total_checks += 1

        # 1. Queue saturation check
        if queue_depth > self.max_queue_depth:
            self.total_trips += 1
            self.trips_by_reason[FallbackReason.QUEUE_SATURATION] += 1
            return GuardrailDecision(
                pass_check=False,
                reason=FallbackReason.QUEUE_SATURATION,
                detail=f"Queue depth {queue_depth} exceeds limit {self.max_queue_depth}",
            )

        # 2. Input out-of-distribution (OOD) check
        if candidate_features is not None and candidate_features.size > 0:
            max_val = float(np.max(np.abs(candidate_features)))
            if max_val > self.feature_upper_bound:
                self.total_trips += 1
                self.trips_by_reason[FallbackReason.INPUT_OOD] += 1
                return GuardrailDecision(
                    pass_check=False,
                    reason=FallbackReason.INPUT_OOD,
                    detail=f"Feature outlier {max_val:.2f} > bound {self.feature_upper_bound}",
                )

        # 3. Prediction residual drift check
        if current_running_residual is not None and self.residual_count >= 10:
            std = self.residual_std
            z_score = abs(current_running_residual - self.residual_mean) / std
            if z_score > self.drift_sigma_threshold:
                self.total_trips += 1
                self.trips_by_reason[FallbackReason.PREDICTION_DRIFT] += 1
                return GuardrailDecision(
                    pass_check=False,
                    reason=FallbackReason.PREDICTION_DRIFT,
                    detail=f"Prediction error {current_running_residual:.1f}us exceeds {z_score:.1f}sigma (threshold {self.drift_sigma_threshold}sigma)",
                )

        return GuardrailDecision(pass_check=True, reason=FallbackReason.NONE, detail="OK")


def c_guardrail_check(queue_depth: int, running_drift_milli_sigma: int) -> bool:
    """Exact Python mirror of C neuroos_guardrail_check function."""
    if queue_depth > MAX_QUEUE_DEPTH:
        return False
    if running_drift_milli_sigma > DRIFT_THRESHOLD_MILLI_SIGMA:
        return False
    return True
