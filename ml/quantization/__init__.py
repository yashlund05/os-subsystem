"""Quantization package init.

.. deprecated::
    DEPRECATED MODULE (remediation item 1.4) — 2026-10-08

    ``ml/quantization/`` is an unreferenced reimplementation that is NOT called
    from any live code path or kernel test.  The active, unit-tested
    implementation is in ``quantization/`` (top-level directory).

    Do NOT edit this module expecting changes to take effect in the kernel fast
    path.  Edit ``quantization/int8_forward.c`` / ``quantization/int8_forward.py``
    instead.

    This module will be deleted in a future cleanup pass.
"""

import warnings

from ml.quantization.lut import QuantumLUT, build_quantum_lut
from ml.quantization.quantize import (
    c_header_bytes,
    quantize_student,
    quantized_forward_int,
    save_quantized_policy,
)

warnings.warn(
    "ml.quantization is DEPRECATED and unreferenced. "
    "Use the top-level 'quantization/' package instead. "
    "(Remediation item 1.4 — will be removed in a future cleanup.)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "quantize_student",
    "quantized_forward_int",
    "save_quantized_policy",
    "c_header_bytes",
    "QuantumLUT",
    "build_quantum_lut",
]
