"""Export utility for NeuroOS candidate scorer policy.

Dumps:
- Actor weights in npz and JSON formats with explicit layer shapes
- Exact feature ordering matching observation.py
- Normalization constants (10 task-specific and 6 global features)
- Standalone NumPy-based inference engine for verification
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ml.training.policy import FEATURE_NAMES, CandidateScorer

# Normalization constants used by ObservationEncoder
NORMALIZATION_CONSTANTS: Dict[str, float] = {
    # 10 task feature scalers
    "max_burst_us": 100000.0,
    "max_wait_us": 500000.0,
    "max_ctx_switches": 50.0,
    "max_pmu_delta": 5000.0,
    "max_mem_kb": 65536.0,
    "max_priority": 10.0,
    # 6 global feature scalers
    "max_queue_depth": 1024.0,
    "max_load_factor": 2.0,
    "max_cpu_busy": 1.0,
    "max_global_wait_us": 500000.0,
    "max_global_burst_us": 100000.0,
    "max_switch_delta_us": 100000.0,
}


def extract_actor_parameters(scorer: CandidateScorer) -> Dict[str, np.ndarray]:
    """Extracts weights and biases from the CandidateScorer into numpy arrays."""
    params: Dict[str, np.ndarray] = {}
    linear_idx = 0
    for module in scorer.net:
        if isinstance(module, torch.nn.Linear):
            params[f"layer_{linear_idx}_weight"] = module.weight.detach().cpu().numpy()
            params[f"layer_{linear_idx}_bias"] = module.bias.detach().cpu().numpy()
            linear_idx += 1
    return params


def numpy_scorer_forward(
    params: Dict[str, np.ndarray], candidates: np.ndarray, action_mask: np.ndarray
) -> np.ndarray:
    """
    Pure NumPy implementation of CandidateScorer with masked softmax.
    Matches PyTorch forward pass within numerical tolerance (< 1e-5).

    Args:
        params: Extracted weights and biases
        candidates: Array of shape (top_k, 16) or (batch, top_k, 16)
        action_mask: Array of shape (top_k,) or (batch, top_k)
    Returns:
        probabilities: Masked softmax probabilities across top_k
    """
    is_batched = candidates.ndim == 3
    if not is_batched:
        candidates = np.expand_dims(candidates, axis=0)
        action_mask = np.expand_dims(action_mask, axis=0)

    b, k, d = candidates.shape
    x = candidates.reshape(b * k, d)

    # Forward through layers
    num_layers = len([k for k in params.keys() if "weight" in k])
    for l_idx in range(num_layers):
        w = params[f"layer_{l_idx}_weight"]  # (out_dim, in_dim)
        bias = params[f"layer_{l_idx}_bias"]  # (out_dim,)
        x = np.dot(x, w.T) + bias
        if l_idx < num_layers - 1:
            x = np.maximum(0.0, x)  # ReLU

    raw_scores = x.reshape(b, k)

    # Masked softmax: set invalid slots to very negative number
    masked_scores = np.where(action_mask > 0, raw_scores, -1e9)
    # Numerical stability shift
    max_scores = np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(masked_scores - max_scores) * (action_mask > 0)
    denom = np.sum(exp_scores, axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    probs = exp_scores / denom

    if not is_batched:
        return probs[0]
    return probs


def export_policy_for_quantization(
    scorer: CandidateScorer,
    export_dir: str,
    prefix: str = "neuroos_scorer",
) -> Tuple[Path, Path]:
    """
    Exports actor weights, shapes, feature order, and normalization statistics.

    Saves:
        1. <export_dir>/<prefix>_weights.npz
        2. <export_dir>/<prefix>_metadata.json
    """
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = extract_actor_parameters(scorer)

    npz_path = out_dir / f"{prefix}_weights.npz"
    json_path = out_dir / f"{prefix}_metadata.json"

    # Save weights
    np.savez(npz_path, **params)

    # Extract layer architectures
    layers_meta: List[Dict[str, Any]] = []
    linear_idx = 0
    for module in scorer.net:
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().cpu().numpy()
            layers_meta.append(
                {
                    "layer_index": linear_idx,
                    "in_features": int(w.shape[1]),
                    "out_features": int(w.shape[0]),
                    "weight_shape": list(w.shape),
                    "bias_shape": list(module.bias.shape),
                }
            )
            linear_idx += 1

    metadata = {
        "model_type": "CandidateScorer",
        "input_dim": scorer.input_dim,
        "hidden_dims": scorer.hidden_dims,
        "num_layers": linear_idx,
        "feature_order": FEATURE_NAMES,
        "feature_slices": {
            "task_features": {"start": 0, "end": 10, "dim": 10},
            "global_features": {"start": 10, "end": 16, "dim": 6},
        },
        "normalization_constants": NORMALIZATION_CONSTANTS,
        "layers": layers_meta,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path, json_path
