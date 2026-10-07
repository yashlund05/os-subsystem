"""User-space distillation daemon package (Phase 2 Week 4).

Thin daemon wrappers over ml.distillation / ml.quantization cores.
Separation per docs/Rules.md Rule 9: prototype training in ml/, daemon in userspace/.

.. deprecated::
    DEPRECATED MODULE (remediation item 1.4) — 2026-10-08

    ``userspace/distillation/`` is an unreferenced reimplementation that is NOT
    called from any live code path or test.  The active, tested implementation
    lives in ``quantization/`` (top-level directory).

    Do NOT edit this module expecting changes to take effect.  Edit
    ``quantization/int8_forward.c`` or ``quantization/int8_forward.py`` instead.

    This module will be deleted in a future cleanup pass.
"""

import warnings

from userspace.distillation.distiller import run_distillation_job
from userspace.distillation.lut import build_kernel_tables
from userspace.distillation.quantize import quantize_and_export

warnings.warn(
    "userspace.distillation is DEPRECATED and unreferenced. "
    "Use the top-level 'quantization/' package instead. "
    "(Remediation item 1.4 — will be removed in a future cleanup.)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["run_distillation_job", "quantize_and_export", "build_kernel_tables"]

