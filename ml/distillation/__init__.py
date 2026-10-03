"""Teacher-student distillation package init."""

from ml.distillation.distiller import (
    DistillationConfig,
    distill_student_from_teacher,
    numpy_student_forward,
)

__all__ = ["DistillationConfig", "distill_student_from_teacher", "numpy_student_forward"]
