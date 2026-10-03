"""Python-layer guardrail evaluator (mirrors kernel/guardrails/guardrail.{h,c}).

Used by:
- tests/integration/test_integration_pipeline.py
- schedulers/neuroos_lite/scheduler.py (already inlined, but exposed here for
  standalone unit and integration testing)

Mirrors the O(1) deterministic guardrail logic from guardrail.c:
  - GUARDRAIL A: queue depth > threshold  → fallback required
  - GUARDRAIL B: prediction error > sigma_threshold → fallback required
"""

from __future__ import annotations


class Guardrail:
    """Deterministic O(1) guardrail evaluator.

    Mirrors ``neuroos_guardrail_eval`` from ``kernel/guardrails/guardrail.c``.

    Args:
        queue_depth_threshold:  Fallback is triggered when queue depth exceeds
                                this value (default 1024, per TRD §6).
        drift_sigma_threshold:  Fallback is triggered when prediction error
                                exceeds this many standard deviations (default 3.0).
    """

    def __init__(
        self,
        queue_depth_threshold: int = 1024,
        drift_sigma_threshold: float = 3.0,
    ) -> None:
        self.queue_depth_threshold = queue_depth_threshold
        self.drift_sigma_threshold = drift_sigma_threshold
        self._last_reason: str = "none"

    def check(self, queue_depth: int, prediction_error_sigma: float) -> bool:
        """Evaluate guardrails.

        Returns:
            True  → fallback (MLFQ) is required.
            False → neural path is safe.
        """
        if queue_depth > self.queue_depth_threshold:
            self._last_reason = "queue_saturation"
            return True
        if abs(prediction_error_sigma) > self.drift_sigma_threshold:
            self._last_reason = "drift_trip"
            return True
        self._last_reason = "none"
        return False

    @property
    def last_reason(self) -> str:
        """Human-readable reason for the last evaluation result."""
        return self._last_reason

    def __repr__(self) -> str:
        return (
            f"Guardrail(queue_thresh={self.queue_depth_threshold}, "
            f"drift_thresh={self.drift_sigma_threshold})"
        )
