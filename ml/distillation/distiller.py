"""Teacher-student knowledge distillation for NeuroOS scheduling policies.

Phase 2 Week 4 per docs/Phases.md:
- Compress deep teacher (16 -> 64 -> 32 -> 1) into compact student (16 -> 8 -> 1).
- Uses soft-target MSE distillation on teacher logits/scores.
- No oracle burst access: trains purely on observation features (16-D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only CI fallback
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    _TORCH_AVAILABLE = False


@dataclass
class DistillationConfig:
    """Hyperparameters for KD (mirrors configs/model/mlp_16_8_1.json)."""

    alpha_soft: float = 0.7
    temperature: float = 2.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 20
    seed: int = 42


def require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for distillation (pip install neuroos-lite[ml])")


def distill_student_from_teacher(
    teacher: "torch.nn.Module",
    student: "torch.nn.Module",
    features: np.ndarray,
    config: Optional[DistillationConfig] = None,
) -> Dict[str, float]:
    """Distill teacher scores into student via soft-target regression.

    Args:
        teacher: Pretrained CandidateScorer (e.g. 16->64->32->1), in eval mode.
        student: Untrained CandidateScorer (16->8->1).
        features: Array of shape (N, 16) float32 observation vectors.
        config: Distillation hyperparameters.

    Returns:
        Dict with final mse, r2 and epochs run. No checkpoints written here.
    """
    require_torch()
    cfg = config or DistillationConfig()
    rng = np.random.default_rng(cfg.seed)

    assert features.ndim == 2 and features.shape[1] == 16, "features must be (N, 16)"
    teacher.eval()
    student.train()

    with torch.no_grad():
        t_in = torch.from_numpy(features.astype(np.float32))
        # Teacher forward: support (N,16) -> (N,) scores
        t_scores = teacher(t_in).detach().numpy().reshape(-1)

    optimizer = optim.Adam(student.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    n = features.shape[0]

    last_loss = float("inf")
    for _ in range(cfg.num_epochs):
        perm = rng.permutation(n)
        epoch_losses: List[float] = []
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            xb = torch.from_numpy(features[idx].astype(np.float32))
            yb = torch.from_numpy(t_scores[idx].astype(np.float32))
            pred = student(xb).reshape(-1)
            # Temperature-scaled soft regression (linear scores, so plain MSE)
            loss = loss_fn(pred / cfg.temperature, yb / cfg.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        last_loss = float(np.mean(epoch_losses))

    # Fidelity: R^2 of student vs teacher on full set
    student.eval()
    with torch.no_grad():
        s_scores = student(torch.from_numpy(features.astype(np.float32))).detach().numpy().reshape(-1)
    ss_res = float(np.sum((t_scores - s_scores) ** 2))
    ss_tot = float(np.sum((t_scores - np.mean(t_scores)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    mse = float(np.mean((t_scores - s_scores) ** 2))
    return {"mse": mse, "r2": r2, "final_loss": last_loss, "epochs": float(cfg.num_epochs)}


def numpy_student_forward(params: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Pure-NumPy forward for 16->8->1 student (verification, no torch).

    params keys: layer_0_weight (8,16), layer_0_bias (8,), layer_1_weight (1,8), layer_1_bias (1,).
    """
    w1 = params["layer_0_weight"]
    b1 = params["layer_0_bias"]
    w2 = params["layer_1_weight"]
    b2 = params["layer_1_bias"]
    h = np.maximum(0.0, x @ w1.T + b1)
    out = h @ w2.T + b2
    return out.reshape(-1)
