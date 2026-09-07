"""Common benchmark result structure for experimental runs."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SubsystemType(str, Enum):
    SCHEDULING = "scheduling"
    MEMORY = "memory"
    OVERHEAD = "overhead"
    ABLATION = "ablation"


@dataclass
class BenchmarkResult:
    """
    Standard benchmark result representation across all experiments.
    Matches schema in docs/Schema.md.
    """
    experiment_id: str
    subsystem: SubsystemType
    algorithm: str
    workload_type: str
    load_factor: float
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    guardrail_fallback_trips: int = 0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "subsystem": self.subsystem.value,
            "algorithm": self.algorithm,
            "workload_type": self.workload_type,
            "load_factor": self.load_factor,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "guardrail_fallback_trips": self.guardrail_fallback_trips,
            "notes": self.notes
        }
