"""Abstract policy model interface supporting FP32, Quantized INT8, LUT, and Tiny MLP."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence


class ModelType(str, Enum):
    FP32_REFERENCE = "FP32_REFERENCE"
    QUANTIZED_INT8 = "QUANTIZED_INT8"
    TINY_MLP = "TINY_MLP"
    LOOKUP_TABLE = "LOOKUP_TABLE"


@dataclass
class PolicyDecision:
    """Output decision from a policy evaluation."""
    task_index: int
    dynamic_quantum_us: int
    priority_score: float
    confidence: float = 1.0


class PolicyModel(ABC):
    """
    Abstract base class for all uniprocessor scheduling and allocation policies.
    """
    def __init__(self, model_type: ModelType, input_dim: int = 16) -> None:
        self.model_type = model_type
        self.input_dim = input_dim

    @abstractmethod
    def evaluate(self, feature_vectors: Sequence[Sequence[float]]) -> Sequence[PolicyDecision]:
        """
        Evaluate features for a candidate list of tasks and return prioritization decisions.
        """
        pass

    @abstractmethod
    def serialize_weights(self) -> bytes:
        """
        Serialize model weights into the target binary representation.
        """
        pass


class DummyReferencePolicy(PolicyModel):
    """
    Placeholder/Scaffold policy implementing Shortest Remaining Time First heuristic
    as a baseline until Phase 2 DRL is integrated.
    """
    def __init__(self) -> None:
        super().__init__(model_type=ModelType.FP32_REFERENCE, input_dim=16)

    def evaluate(self, feature_vectors: Sequence[Sequence[float]]) -> Sequence[PolicyDecision]:
        decisions: List[PolicyDecision] = []
        for idx, feat in enumerate(feature_vectors):
            # feat[0] represents remaining burst in normalized features
            rem_burst = feat[0] if len(feat) > 0 else 1.0
            score = 1.0 / (rem_burst + 1e-5)
            decisions.append(
                PolicyDecision(
                    task_index=idx,
                    dynamic_quantum_us=5000,
                    priority_score=score
                )
            )
        return decisions

    def serialize_weights(self) -> bytes:
        # 16 -> 8 -> 1 weights placeholder
        return b"\x00" * 156
