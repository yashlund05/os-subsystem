"""Training CLI and engine for NeuroOS PPO policy."""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from ml.training.export import export_policy_for_quantization
from ml.training.policy import ScorerPolicy
from ml.training.ppo import PPOConfig, PPOTrainer
from ml.training.vec_env import VectorSchedulerEnv
from simulator.scheduling.task import SimulatedTask
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv
from userspace.trainer.reward import RewardConfig


def get_git_commit_hash() -> str:
    """Returns the current git commit hash or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def build_workload_generator(
    workload_type: str,
    num_tasks: int,
    load_factor: float,
    alpha: float = 1.3,
    seed_offset: int = 1000,
) -> Callable[[int], List[SimulatedTask]]:
    """Constructs a deterministic task generator based on episode seed."""

    def generator(episode_seed: int) -> List[SimulatedTask]:
        effective_seed = (seed_offset + episode_seed) % 1000000
        if workload_type == "convoy":
            return AdversarialWorkloadGenerator.create_convoy_workload(
                num_short_jobs=num_tasks - 1, long_burst_us=50000, short_burst_us=100
            )
        else:
            synth = SyntheticWorkloadGenerator(seed=effective_seed)
            return synth.generate_pareto_bursts(
                num_tasks=num_tasks,
                alpha=alpha,
                min_burst_us=200,
                load_factor=load_factor,
            )

    return generator


def make_env_factory(
    workload_type: str,
    num_tasks: int,
    load_factor: float,
    alpha: float,
    top_k: int,
    reward_config: RewardConfig,
    seed_offset: int,
) -> Callable[[], SchedulerEnv]:
    """Returns an env factory closure for VectorSchedulerEnv."""
    gen = build_workload_generator(
        workload_type=workload_type,
        num_tasks=num_tasks,
        load_factor=load_factor,
        alpha=alpha,
        seed_offset=seed_offset,
    )

    def _init() -> SchedulerEnv:
        return SchedulerEnv(
            workload_generator=gen,
            top_k=top_k,
            load_factor=load_factor,
            reward_config=reward_config,
        )

    return _init


def train_single_run(
    cfg: Dict[str, Any],
    train_seed: int,
    device: torch.device,
    actor_hidden_dims: List[int],
    curriculum_mode: str,
    run_name: str,
) -> Tuple[ScorerPolicy, Dict[str, Any]]:
    """Executes a single PPO training run for a given seed and config."""
    # Set seeds
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_seed)

    env_cfg = cfg["environment"]
    rew_cfg = cfg["reward"]
    ppo_cfg_dict = cfg["ppo"]
    cur_cfg = cfg["curriculum"]

    reward_config = RewardConfig(
        w_wait=rew_cfg.get("w_wait", rew_cfg.get("step_wait_penalty_weight", 1.0)),
        w_completion=rew_cfg.get("w_completion", rew_cfg.get("completion_bonus", 0.0)),
        w_switch=rew_cfg.get("w_switch", rew_cfg.get("context_switch_penalty_weight", 0.02)),
        w_starvation=rew_cfg.get("w_starvation", rew_cfg.get("max_wait_penalty_weight", 0.1)),
        w_tail_threshold=rew_cfg.get("w_tail_threshold", rew_cfg.get("tail_penalty_weight", 0.1)),
    )

    ppo_config = PPOConfig(
        learning_rate=ppo_cfg_dict["learning_rate"],
        gamma=ppo_cfg_dict["gamma"],
        gae_lambda=ppo_cfg_dict["gae_lambda"],
        clip_epsilon=ppo_cfg_dict["clip_epsilon"],
        value_coef=ppo_cfg_dict["value_coef"],
        entropy_coef_start=ppo_cfg_dict["entropy_coef_start"],
        entropy_coef_end=ppo_cfg_dict["entropy_coef_end"],
        max_grad_norm=ppo_cfg_dict["max_grad_norm"],
        num_rollout_steps=ppo_cfg_dict["num_rollout_steps"],
        num_epochs=ppo_cfg_dict["num_epochs"],
        batch_size=ppo_cfg_dict["batch_size"],
        target_kl=ppo_cfg_dict.get("target_kl", 0.02),
    )

    # Instantiate policy
    policy = ScorerPolicy(
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=cfg["network"].get("critic_hidden_dims", [64, 64]),
    )

    # Build vector environment
    num_envs = env_cfg["num_envs"]
    env_fns = [
        make_env_factory(
            workload_type="pareto",
            num_tasks=env_cfg["num_tasks"],
            load_factor=env_cfg["load_factor"],
            alpha=1.3,
            top_k=env_cfg["top_k"],
            reward_config=reward_config,
            seed_offset=train_seed * 1000 + i * 100,
        )
        for i in range(num_envs)
    ]
    vec_env = VectorSchedulerEnv(env_fns)

    trainer = PPOTrainer(policy=policy, vec_env=vec_env, config=ppo_config, device=device)

    total_timesteps_target = cur_cfg["total_timesteps"]

    obs_cands, obs_masks = vec_env.reset()

    logs: List[Dict[str, Any]] = []
    start_time = time.time()
    ep_returns_window: List[float] = []

    print(
        f"[{run_name}] Starting training: target={total_timesteps_target} steps, device={device}, seed={train_seed}"
    )

    while trainer.total_timesteps < total_timesteps_target:
        progress = float(trainer.total_timesteps) / float(max(1, total_timesteps_target))

        buffers, obs_cands, obs_masks, completed_returns = trainer.collect_rollouts(
            obs_cands, obs_masks
        )
        ep_returns_window.extend(completed_returns)

        train_metrics = trainer.update(buffers, progress=progress)

        mean_ret = float(np.mean(ep_returns_window[-20:])) if ep_returns_window else 0.0
        elapsed_sec = time.time() - start_time

        log_entry = {
            "timesteps": trainer.total_timesteps,
            "progress": round(progress, 4),
            "mean_return": round(mean_ret, 3),
            "policy_loss": round(train_metrics["policy_loss"], 5),
            "value_loss": round(train_metrics["value_loss"], 5),
            "entropy": round(train_metrics["entropy"], 4),
            "approx_kl": round(train_metrics["approx_kl"], 5),
            "elapsed_seconds": round(elapsed_sec, 2),
        }
        logs.append(log_entry)

        if len(logs) % 5 == 0 or trainer.total_timesteps >= total_timesteps_target:
            print(
                f"[{run_name}] Step {trainer.total_timesteps}/{total_timesteps_target} "
                f"({progress * 100:.1f}%) | Ret: {mean_ret:.2f} | "
                f"Ploss: {train_metrics['policy_loss']:.4f} | Vloss: {train_metrics['value_loss']:.4f} | "
                f"Entropy: {train_metrics['entropy']:.3f} | Elapsed: {elapsed_sec:.1f}s"
            )

    vec_env.close()
    wall_clock = time.time() - start_time

    # Assess convergence: compare first quarter vs last quarter mean returns
    converged = False
    if len(ep_returns_window) >= 10:
        quarter = max(2, len(ep_returns_window) // 4)
        early_perf = np.mean(ep_returns_window[:quarter])
        late_perf = np.mean(ep_returns_window[-quarter:])
        converged = bool(late_perf >= early_perf - 5.0)

    summary = {
        "run_name": run_name,
        "seed": train_seed,
        "total_timesteps": trainer.total_timesteps,
        "wall_clock_seconds": round(wall_clock, 2),
        "converged": converged,
        "final_mean_return": round(
            float(np.mean(ep_returns_window[-30:])) if ep_returns_window else 0.0, 3
        ),
        "logs": logs,
    }

    return policy, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuroOS PPO Scheduling Policy")
    parser.add_argument(
        "--config", type=str, default="configs/ppo_quick.yaml", help="Path to YAML config"
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda override")
    parser.add_argument("--curriculum", type=str, default=None, help="mixed or staged override")
    parser.add_argument(
        "--architecture", type=str, default=None, help="teacher or student override"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determine device
    device_str = args.device or cfg["experiment"].get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Architecture selection
    arch = args.architecture or cfg["network"].get("architecture", "teacher")
    if arch == "student":
        actor_hidden = [8]
    else:
        actor_hidden = [64, 32]

    curriculum_mode = args.curriculum or cfg["curriculum"].get("mode", "mixed")
    exp_name = cfg["experiment"].get("name", "ppo_run")
    base_seed = cfg["experiment"].get("seed", 1001)
    num_seeds = cfg["experiment"].get("num_training_seeds", 1)
    seeds = [base_seed + i for i in range(num_seeds)]

    export_dir = Path(cfg["experiment"].get("export_dir", "ml/checkpoints"))
    export_dir.mkdir(parents=True, exist_ok=True)

    git_hash = get_git_commit_hash()
    print("============================================================")
    print("NeuroOS DRL PPO Training Pipeline")
    print(f"Config: {args.config} | Architecture: {arch} ({actor_hidden})")
    print(f"Curriculum: {curriculum_mode} | Device: {device} | Git: {git_hash}")
    print(f"Seeds: {seeds}")
    print("============================================================")

    all_summaries = []
    best_policy: Optional[ScorerPolicy] = None
    best_return = -1e9

    for s in seeds:
        run_name = f"{exp_name}_{arch}_{curriculum_mode}_s{s}"
        policy, summary = train_single_run(
            cfg=cfg,
            train_seed=s,
            device=device,
            actor_hidden_dims=actor_hidden,
            curriculum_mode=curriculum_mode,
            run_name=run_name,
        )
        all_summaries.append(summary)

        # Checkpoint per seed
        ckpt_path = export_dir / f"{run_name}_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "actor_hidden_dims": actor_hidden,
                "seed": s,
                "git_hash": git_hash,
                "config": cfg,
                "summary": summary,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        if summary["final_mean_return"] > best_return:
            best_return = summary["final_mean_return"]
            best_policy = policy

    # Export best policy for Quantization Lead (npz + JSON)
    if best_policy is not None:
        prefix = f"neuroos_{arch}_{curriculum_mode}"
        npz_p, json_p = export_policy_for_quantization(
            best_policy.actor, str(export_dir), prefix=prefix
        )
        print(f"Exported quantization package:\n  Weights: {npz_p}\n  Metadata: {json_p}")

    # Save summary manifest
    manifest_path = export_dir / f"{exp_name}_{arch}_{curriculum_mode}_summary.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": exp_name,
                "architecture": arch,
                "curriculum_mode": curriculum_mode,
                "git_hash": git_hash,
                "runs": all_summaries,
            },
            f,
            indent=2,
        )
    print(f"Saved run summary to {manifest_path}")


if __name__ == "__main__":
    main()
