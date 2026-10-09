"""Teacher-student distillation package init.

.. deprecated::
    DEPRECATED MODULE (remediation item 1.4) — 2026-10-08

    ``ml/distillation/`` is an unreferenced reimplementation that is NOT called
    from any live code path or test.  The active, tested implementation lives in
    ``quantization/`` (top-level).

    Do NOT edit this module expecting changes to take effect in the kernel fast
    path.  If you need distillation changes, edit ``quantization/int8_forward.c``
    or ``quantization/int8_forward.py`` instead.

    This module will be deleted in a future cleanup pass.
"""

import warnings

from ml.distillation.distiller import (
    DistillationConfig,
    distill_student_from_teacher,
    numpy_student_forward,
)

warnings.warn(
    "ml.distillation is DEPRECATED and unreferenced. "
    "Use the top-level 'quantization/' package instead. "
    "(Remediation item 1.4 — will be removed in a future cleanup.)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DistillationConfig", "distill_student_from_teacher", "numpy_student_forward"]
