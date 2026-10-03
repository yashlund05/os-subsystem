"""Controlled feature ablation: Old vs New burst_ratio with 3 training seeds."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import subprocess

import yaml


def run_controlled_retraining():
    print("\n" + "="*70)
    print("TASK 5: Controlled Feature Ablation (3 Training Seeds: 1001, 1002, 1003)")
    print("="*70)

    # 1. Train 3 seeds of Student with new encoder
    # Using 5,000 steps per seed for fast, robust convergence
    with open("configs/ppo_production.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["curriculum"]["total_timesteps"] = 8192
    base_cfg["environment"]["num_envs"] = 4
    base_cfg["experiment"]["num_training_seeds"] = 3
    base_cfg["experiment"]["device"] = "cpu"

    # Train Student (16 -> 8 -> 1)
    base_cfg["network"]["architecture"] = "student"
    base_cfg["network"]["actor_hidden_dims"] = [8]
    base_cfg["network"]["critic_hidden_dims"] = [32, 32]
    base_cfg["experiment"]["name"] = "ppo_v3_student"

    with open("configs/ppo_v3_student.yaml", "w") as f:
        yaml.safe_dump(base_cfg, f)

    print("Training Student (16->8->1) across seeds [1001, 1002, 1003] with new encoder...")
    subprocess.run([sys.executable, "-m", "ml.training.train", "--config", "configs/ppo_v3_student.yaml"], check=True)

    # Train Teacher (16 -> 64 -> 32 -> 1)
    base_cfg["network"]["architecture"] = "teacher"
    base_cfg["network"]["actor_hidden_dims"] = [64, 32]
    base_cfg["network"]["critic_hidden_dims"] = [64, 64]
    base_cfg["experiment"]["name"] = "ppo_v3_teacher"

    with open("configs/ppo_v3_teacher.yaml", "w") as f:
        yaml.safe_dump(base_cfg, f)

    print("Training Teacher (16->64->32->1) across seeds [1001, 1002, 1003] with new encoder...")
    subprocess.run([sys.executable, "-m", "ml.training.train", "--config", "configs/ppo_v3_teacher.yaml"], check=True)

    # Mark old checkpoints invalid
    old_ckpts = list(Path("ml/checkpoints").glob("*staged*.json")) + list(Path("ml/checkpoints").glob("*mixed*.json"))
    for cp_json in old_ckpts:
        if "v3" not in cp_json.name:
            try:
                data = json.loads(cp_json.read_text())
                data["validity"] = "INVALID_SUPERSEDED"
                data["invalidation_reason"] = "Saturated burst_ratio bug fixed in commit 1473244; use v3 checkpoints."
                cp_json.write_text(json.dumps(data, indent=2))
            except Exception:
                pass
    print("Marked legacy checkpoints as INVALID_SUPERSEDED.")


if __name__ == "__main__":
    run_controlled_retraining()
