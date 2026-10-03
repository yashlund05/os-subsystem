"""Daemon quantization entry point (wraps ml.quantization)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ml.quantization.quantize import quantize_student, save_quantized_policy


def quantize_and_export(
    student_weights_npz: str, export_dir: str, prefix: str = "neuroos_student", version: int = 1
) -> Dict[str, Any]:
    data = np.load(student_weights_npz)
    params = {
        "layer_0_weight": np.asarray(data["layer_0_weight"]),
        "layer_0_bias": np.asarray(data["layer_0_bias"]),
        "layer_1_weight": np.asarray(data["layer_1_weight"]),
        "layer_1_bias": np.asarray(data["layer_1_bias"]),
    }
    qpolicy, meta = quantize_student(params, version=version)
    npz_p, json_p = save_quantized_policy(qpolicy, meta, export_dir, prefix=prefix)
    return {"npz": str(npz_p), "json": str(json_p), "checksum": meta["checksum"], "version": version}
