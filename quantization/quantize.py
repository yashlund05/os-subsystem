"""Quantization and calibration pipeline for NeuroOS 16->8->1 Student policy.

Calibrates on real rollout observations from SchedulerEnv, quantizes weights into
symmetric int8, int16 biases, and int32 accumulators, and exports to C header & JSON.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for unit CI
    torch = None  # type: ignore[assignment]

INPUT_DIM = 16
HIDDEN_DIM = 8
OUTPUT_DIM = 1


def collect_calibration_features(
    num_samples: int = 15000,
    seeds: Optional[List[int]] = None,
) -> np.ndarray:
    """Collects real candidate observation vectors from SchedulerEnv rollouts."""
    if seeds is None:
        seeds = list(range(1001, 1025))

    workloads = [
        lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.5),
        lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8),
        lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.8),
        lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100),
        lambda s: SyntheticWorkloadGenerator(seed=s).generate_multiburst_process_workload(
            10, 10, 1.3, 200
        ),
    ]

    features_collected: List[np.ndarray] = []

    for s in seeds:
        wl_fn = workloads[s % len(workloads)]
        env = SchedulerEnv(workload_generator=wl_fn, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            for idx in valid:
                features_collected.append(obs["candidates"][idx])
                if len(features_collected) >= num_samples:
                    return np.array(features_collected, dtype=np.float32)
            act = int(valid[0])
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc

    return np.array(features_collected, dtype=np.float32)


def calibrate_input_scale(calibration_features: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """Calculates per-feature scale vector s_x of shape (16,) using percentile calibration."""
    s_x = np.zeros(16, dtype=np.float64)
    for i in range(16):
        val = float(np.percentile(np.abs(calibration_features[:, i]), percentile))
        s_x[i] = max(1e-5, val) / 127.0
    return s_x


def quantize_student_policy(
    float_weights: Dict[str, np.ndarray],
    s_x: np.ndarray,
    version: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Quantizes float weights (w1, b1, w2, b2) into symmetric integer policy.

    Folds per-feature scale s_x into w1 so that integer inference in C requires
    zero feature-weight rescaling and uses exact row-major memory layout.
    """
    w1 = float_weights["w1"].astype(np.float64)  # (8, 16)
    b1 = float_weights["b1"].astype(np.float64)  # (8,)
    w2 = float_weights["w2"].astype(np.float64)  # (1, 8)
    b2 = float_weights["b2"].astype(np.float64)  # (1,)

    assert w1.shape == (8, 16), f"w1 shape {w1.shape} != (8, 16)"
    assert b1.shape == (8,), f"b1 shape {b1.shape} != (8,)"
    assert w2.shape == (1, 8), f"w2 shape {w2.shape} != (1, 8)"
    assert s_x.shape == (16,), f"s_x shape {s_x.shape} != (16,)"

    # Fold input scales into w1: w1_eff[j, i] = w1[j, i] * s_x[i]
    w1_eff = w1 * s_x.reshape(1, 16)

    # Scale for w1_eff
    s_w1 = float(np.max(np.abs(w1_eff)) / 127.0)
    q_w1 = np.clip(np.round(w1_eff / s_w1), -128, 127).astype(np.int8)

    # b1 scale is s_w1 (since input accumulator is sum_i q_x,i * q_w1,j,i)
    s_b1 = s_w1
    q_b1 = np.clip(np.round(b1 / s_b1), -32768, 32767).astype(np.int16)

    # w2 scale
    s_w2 = float(np.max(np.abs(w2)) / 127.0)
    q_w2 = np.clip(np.round(w2 / s_w2), -128, 127).astype(np.int8)

    # b2 scale is s_w1 * s_w2
    s_b2 = float(s_w1 * s_w2)
    q_b2 = np.clip(np.round(b2 / s_b2), -2147483648, 2147483647).astype(np.int32)

    # CRC32 over binary image
    crc = zlib.crc32(q_w1.tobytes())
    crc = zlib.crc32(q_b1.tobytes(), crc)
    crc = zlib.crc32(q_w2.tobytes(), crc)
    crc = zlib.crc32(q_b2.tobytes(), crc)
    crc &= 0xFFFFFFFF

    qpolicy = {
        "w1": q_w1,
        "b1": q_b1,
        "w2": q_w2,
        "b2": q_b2,
    }

    meta = {
        "architecture": "16->8->1",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "version": int(version),
        "checksum": int(crc),
        "scales": {
            "s_x": [float(x) for x in s_x],
            "s_w1": s_w1,
            "s_b1": s_b1,
            "s_w2": s_w2,
            "s_b2": s_b2,
            "output_scale": s_b2,
        },
    }

    return qpolicy, meta


def export_c_weights_header(
    qpolicy: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_path: str = "kernel/include/neuroos_weights.h",
) -> Path:
    """Exports quantized weights as a C header struct conforming to neuroos_kernel.h."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w1 = qpolicy["w1"]  # (8, 16)
    b1 = qpolicy["b1"]  # (8,)
    w2 = qpolicy["w2"]  # (1, 8)
    b2 = qpolicy["b2"]  # (1,)

    lines = [
        "#ifndef NEUROOS_WEIGHTS_H",
        "#define NEUROOS_WEIGHTS_H",
        "",
        '#include "neuroos_kernel.h"',
        "",
        f"/* Quantized NeuroOS-Lite Student Policy (Version {meta['version']}, CRC32 0x{meta['checksum']:08X}) */",
        f"/* Input Scale s_x: per-feature (16) array, Output Scale s_b2: {meta['scales']['s_b2']:.10f} */",
        "",
        "static const struct neuroos_quantized_policy neuroos_student_policy = {",
        "    /* w1: (8 x 16) row-major */",
        "    .w1 = {",
    ]

    for j in range(8):
        row_vals = [f"{int(w1[j, i])}" for i in range(16)]
        comma = "," if j < 7 else ""
        lines.append(f"        /* neuron {j} */ " + ", ".join(row_vals) + comma)

    s_x_vals = ", ".join(f"{float(x):.8f}f" for x in meta["scales"]["s_x"])

    lines.extend(
        [
            "    },",
            "    /* b1: (8) int16 */",
            "    .b1 = {",
            "        " + ", ".join(str(int(b)) for b in b1),
            "    },",
            "    /* w2: (1 x 8) int8 */",
            "    .w2 = {",
            "        " + ", ".join(str(int(w)) for w in w2.reshape(-1)),
            "    },",
            "    /* b2: (1) int32 */",
            f"    .b2 = {{ {int(b2[0])} }},",
            f"    .version = {meta['version']}U,",
            "};",
            "",
            f"static const float neuroos_scale_s_x[16] = {{ {s_x_vals} }};",
            f"#define NEUROOS_SCALE_S_W1    {meta['scales']['s_w1']}f",
            f"#define NEUROOS_SCALE_S_B1    {meta['scales']['s_b1']}f",
            f"#define NEUROOS_SCALE_S_W2    {meta['scales']['s_w2']}f",
            f"#define NEUROOS_SCALE_S_B2    {meta['scales']['s_b2']}f",
            "",
            "#endif /* NEUROOS_WEIGHTS_H */",
            "",
        ]
    )

    content = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    return out


def run_quantization_pipeline(
    checkpoint_path: str = "ml/checkpoints/student_bc_ppo_s1003.pt",
    output_dir: str = "ml/checkpoints",
    c_header_path: str = "kernel/include/neuroos_weights.h",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Full quantization pipeline from PyTorch checkpoint to int8 artifacts."""
    if torch is None:
        raise ImportError("torch is required for run_quantization_pipeline but is not installed.")
    print(f"Loading Student checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    float_weights = {
        "w1": state["actor.net.0.weight"].numpy(),
        "b1": state["actor.net.0.bias"].numpy(),
        "w2": state["actor.net.2.weight"].numpy(),
        "b2": state["actor.net.2.bias"].numpy(),
    }

    print("Collecting calibration features from SchedulerEnv rollouts...")
    calib_feats = collect_calibration_features(num_samples=15000)
    print(f"  -> Collected {len(calib_feats)} candidate vectors.")

    s_x = calibrate_input_scale(calib_feats, percentile=98.0)
    print(
        f"  -> Calibrated per-feature input scales s_x (min={s_x.min():.6f}, max={s_x.max():.6f})"
    )

    qpolicy, meta = quantize_student_policy(float_weights, s_x=s_x, version=1)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    json_path = out_p / "quantized_student_int8.json"
    npz_path = out_p / "quantized_student_int8.npz"

    np.savez(
        npz_path,
        w1=qpolicy["w1"],
        b1=qpolicy["b1"],
        w2=qpolicy["w2"],
        b2=qpolicy["b2"],
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    export_c_weights_header(qpolicy, meta, output_path=c_header_path)

    print("Quantization successful! Saved:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ:  {npz_path}")
    print(f"  C Header: {c_header_path}")
    return qpolicy, meta


if __name__ == "__main__":
    run_quantization_pipeline()
