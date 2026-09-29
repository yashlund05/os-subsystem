"""Daemon distillation job: teacher -> student -> quantized policy.

Reads teacher export (npz+metadata from ml.training.export), trains 16->8->1
student by distillation, quantizes to int8 and writes kernel-ready artifacts.
No GPU required; runs on CPU. No fabricated metrics: returns real MSE/R2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ml.distillation.distiller import DistillationConfig, distill_student_from_teacher


def run_distillation_job(
    teacher_weights_npz: str,
    calibration_features_npy: str,
    export_dir: str,
    prefix: str = "neuroos_student",
    epochs: int = 20,
) -> Dict[str, Any]:
    """Run end-to-end distillation + quantization job.

    Args:
        teacher_weights_npz: Path to teacher *_weights.npz (from export_policy_for_quantization).
        calibration_features_npy: Path to (N,16) float32 calibration features.
        export_dir: Output directory for int8 artifacts.
        prefix: Output filename prefix.
        epochs: KD epochs.

    Returns:
        Summary dict with paths + fidelity metrics. Raises on missing torch.
    """
    import torch

    from ml.quantization.quantize import quantize_student, save_quantized_policy
    from ml.training.policy import CandidateScorer

    data = np.load(teacher_weights_npz)
    # Infer teacher hidden dims from layer shapes
    # layer_0_weight: (h0,16), layer_1_weight: (h1,h0), layer_2_weight: (1,h1)
    h0 = int(data["layer_0_weight"].shape[0])
    h1 = int(data["layer_1_weight"].shape[0]) if "layer_1_weight" in data else 0
    teacher_dims = [h0, h1] if h1 else [h0]

    teacher = CandidateScorer(input_dim=16, hidden_dims=teacher_dims)
    # Load weights manually
    with torch.no_grad():
        li = 0
        for module in teacher.net:
            if isinstance(module, torch.nn.Linear):
                w = torch.from_numpy(np.asarray(data[f"layer_{li}_weight"], dtype=np.float32))
                b = torch.from_numpy(np.asarray(data[f"layer_{li}_bias"], dtype=np.float32))
                module.weight.copy_(w)
                module.bias.copy_(b)
                li += 1

    student = CandidateScorer(input_dim=16, hidden_dims=[8])
    feats = np.load(calibration_features_npy)
    assert feats.ndim == 2 and feats.shape[1] == 16

    metrics = distill_student_from_teacher(
        teacher, student, feats, DistillationConfig(num_epochs=epochs)
    )

    # Extract student params and quantize
    params: Dict[str, np.ndarray] = {}
    li = 0
    for module in student.net:
        if isinstance(module, torch.nn.Linear):
            params[f"layer_{li}_weight"] = module.weight.detach().cpu().numpy()
            params[f"layer_{li}_bias"] = module.bias.detach().cpu().numpy()
            li += 1

    qpolicy, meta = quantize_student(params, version=1)
    npz_p, json_p = save_quantized_policy(qpolicy, meta, export_dir, prefix=prefix)
    return {
        "teacher_dims": teacher_dims,
        "mse": metrics["mse"],
        "r2": metrics["r2"],
        "npz": str(npz_p),
        "json": str(json_p),
        "checksum": meta["checksum"],
    }
