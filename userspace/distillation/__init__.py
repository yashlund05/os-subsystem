"""User-space distillation daemon package (Phase 2 Week 4).

Thin daemon wrappers over ml.distillation / ml.quantization cores.
Separation per docs/Rules.md Rule 9: prototype training in ml/, daemon in userspace/.
"""

from userspace.distillation.distiller import run_distillation_job
from userspace.distillation.lut import build_kernel_tables
from userspace.distillation.quantize import quantize_and_export

__all__ = ["run_distillation_job", "quantize_and_export", "build_kernel_tables"]
