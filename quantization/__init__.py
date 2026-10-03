"""NeuroOS-Lite Quantization Subsystem (Phase 4).

Provides:
- quantize: Calibration, symmetric int8 quantization, and C header exporter.
- int8_forward: Integer-only reference forward inference.
- lut_generator: Precomputed Look-Up Table generator for quantum allocation.
- guardrail: Runtime safety monitoring and fallback triggers.
"""

from quantization.guardrail import (
    FallbackReason,
    GuardrailDecision,
    GuardrailEngine,
    c_guardrail_check,
)
from quantization.int8_forward import (
    int8_forward_batch,
    int8_forward_single,
    quantize_features_to_int8,
)
from quantization.lut_generator import QuantumLUT, generate_c_lut_header
from quantization.quantize import (
    calibrate_input_scale,
    collect_calibration_features,
    export_c_weights_header,
    quantize_student_policy,
    run_quantization_pipeline,
)

__all__ = [
    "int8_forward_single",
    "int8_forward_batch",
    "quantize_features_to_int8",
    "QuantumLUT",
    "generate_c_lut_header",
    "GuardrailEngine",
    "GuardrailDecision",
    "FallbackReason",
    "c_guardrail_check",
    "calibrate_input_scale",
    "collect_calibration_features",
    "quantize_student_policy",
    "export_c_weights_header",
    "run_quantization_pipeline",
]
