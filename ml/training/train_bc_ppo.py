"""Behavior-Cloning (BC) initialized PPO training pipeline with eval learning curves."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from ml.training.policy import ScorerPolicy
from ml.training.ppo import PPOConfig, PPOTrainer
from ml.training.vec_env import VectorSchedulerEnv
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv
from userspace.trainer.reward import RewardConfig


def pretrain_actor_bc(
    policy: ScorerPolicy,
    train_seeds: List[int],
    device: torch.device,
    epochs: int = 500,
) -> float:
    """Pretrain policy actor using Behavior Cloning on the heuristic rule with preemption cue."""
    print("Collecting BC pretraining data from heuristic decisions...")
    X_list = []
    y_list = []

    for s in train_seeds:
        wl_gen = (
            (
                lambda seed: SyntheticWorkloadGenerator(seed=seed).generate_pareto_bursts(
                    50, 1.3, 200, 0.8
                )
            )
            if s % 2 == 0
            else (
                lambda seed: SyntheticWorkloadGenerator(
                    seed=seed
                ).generate_multiburst_process_workload(10, 10, 1.3, 200)
            )
        )
        env = SchedulerEnv(workload_generator=wl_gen, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            for idx in valid:
                feat = obs["candidates"][idx]
                X_list.append(feat)
                # Target: -pred_burst_norm + 1.0 * age_norm + 2.0 * preempt_cue
                y_list.append(-feat[1] + 1.0 * feat[2] + 2.0 * feat[9])
            scores = [
                -obs["candidates"][i, 1]
                + 1.0 * obs["candidates"][i, 2]
                + 2.0 * obs["candidates"][i, 9]
                for i in valid
            ]
            act = valid[np.argmax(scores)]
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc

    X = np.array(X_list)
    y = np.array(y_list)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_norm = (y - y_mean) / max(1e-6, y_std)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_norm, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(policy.actor.parameters(), lr=0.01)
    crit = nn.MSELoss()

    policy.actor.to(device)
    for _ in range(epochs):
        pred = policy.actor(X_t).squeeze(-1)
        loss = crit(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pred_raw = pred.detach().cpu().numpy() * y_std + y_mean
    r2 = 1.0 - float(np.sum((y - pred_raw) ** 2) / np.sum((y - y_mean) ** 2))
    print(f"  -> BC pretraining complete: R2={r2:.4f}, final loss={loss.item():.6f}")
    return r2


def evaluate_policy_on_eval_seeds(
    policy: ScorerPolicy,
    eval_seeds: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates policy on fixed eval seeds across Pareto, Poisson, Convoy, and Multi-burst."""
    workloads = {
        "pareto_rho08": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.3, 200, 0.8
        ),
        "poisson_rho08": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.8, 200, 0.8
        ),
        "convoy": lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100),
        "multiburst": lambda s: SyntheticWorkloadGenerator(
            seed=s
        ).generate_multiburst_process_workload(10, 10, 1.3, 200),
    }

    results = {}
    policy.eval()

    for wl_name, gen_fn in workloads.items():
        mwts = []
        for s in eval_seeds:
            env = SchedulerEnv(workload_generator=gen_fn, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                cands = obs["candidates"]
                mask = obs["action_mask"]
                with torch.no_grad():
                    c_t = torch.from_numpy(cands).unsqueeze(0).to(device)
                    m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                    act, _, _ = policy.act(c_t, m_t, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(act.item()))
                done = term or trunc
            mwts.append(env._compute_episode_metrics()["mean_waiting_time_us"])
        results[wl_name] = float(np.mean(mwts))

    policy.train()
    return results


def train_bc_ppo_run(
    arch_name: str,
    actor_hidden_dims: List[int],
    train_seed: int,
    total_steps: int = 100000,
    eval_interval: int = 25000,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Runs BC initialization followed by PPO fine-tuning with eval learning curves."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f"Starting {arch_name} Seed {train_seed} (Steps: {total_steps}, Device: {device})")
    print("=" * 70)

    # Set seeds
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_seed)

    # 1. Instantiate policy
    policy = ScorerPolicy(
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=[64, 64],
    ).to(device)

    # 2. Behavior Cloning Initialization
    bc_r2 = pretrain_actor_bc(
        policy=policy,
        train_seeds=list(range(train_seed, train_seed + 10)),
        device=device,
        epochs=400,
    )

    eval_seeds = list(range(50000, 50010))  # Fixed 10 eval seeds for curve tracking

    # Step 0 eval (post-BC)
    eval_curves: List[Dict[str, Any]] = []
    e0 = evaluate_policy_on_eval_seeds(policy, eval_seeds, device)
    eval_curves.append({"step": 0, **e0})
    print(
        f"  [Step 0 / BC Init] Pareto: {e0['pareto_rho08']:.1f} us | Poisson: {e0['poisson_rho08']:.1f} us | Convoy: {e0['convoy']:.1f} us | MultiBurst: {e0['multiburst']:.1f} us"
    )

    # 3. Setup Vector Scheduler Env with Little's Law Reward
    rew_cfg = RewardConfig(
        w_wait=1.0,
        w_completion=0.0,
        w_switch=0.02,
        w_starvation=0.1,
        w_tail_threshold=0.1,
    )

    ppo_config = PPOConfig(
        learning_rate=0.0001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef_start=0.01,
        entropy_coef_end=0.001,
        max_grad_norm=0.5,
        num_rollout_steps=128,
        num_epochs=4,
        batch_size=128,
        target_kl=0.02,
    )

    num_envs = 8
    env_fns = [
        (
            lambda idx=i: SchedulerEnv(
                workload_generator=(
                    (
                        lambda seed: SyntheticWorkloadGenerator(
                            seed=(train_seed * 1000 + idx * 100 + seed) % 1000000
                        ).generate_pareto_bursts(50, 1.3, 200, 0.8)
                    )
                    if idx % 2 == 0
                    else (
                        lambda seed: SyntheticWorkloadGenerator(
                            seed=(train_seed * 1000 + idx * 100 + seed) % 1000000
                        ).generate_multiburst_process_workload(10, 10, 1.3, 200)
                    )
                ),
                top_k=16,
                reward_config=rew_cfg,
            )
        )
        for i in range(num_envs)
    ]
    vec_env = VectorSchedulerEnv(env_fns)
    trainer = PPOTrainer(policy=policy, vec_env=vec_env, config=ppo_config, device=device)

    obs_cands, obs_masks = vec_env.reset()
    start_time = time.time()
    last_eval_step = 0

    while trainer.total_timesteps < total_steps:
        progress = float(trainer.total_timesteps) / float(max(1, total_steps))
        buffers, obs_cands, obs_masks, completed_returns = trainer.collect_rollouts(
            obs_cands, obs_masks
        )
        train_metrics = trainer.update(buffers, progress=progress)

        if (
            trainer.total_timesteps - last_eval_step >= eval_interval
            or trainer.total_timesteps >= total_steps
        ):
            last_eval_step = trainer.total_timesteps
            ev = evaluate_policy_on_eval_seeds(policy, eval_seeds, device)
            eval_curves.append({"step": trainer.total_timesteps, **ev})
            elapsed = time.time() - start_time
            print(
                f"  [Step {trainer.total_timesteps}/{total_steps}] "
                f"Pareto: {ev['pareto_rho08']:.1f} us | Poisson: {ev['poisson_rho08']:.1f} us | "
                f"Convoy: {ev['convoy']:.1f} us | MultiBurst: {ev['multiburst']:.1f} us | "
                f"Ploss: {train_metrics['policy_loss']:.4f} | Vloss: {train_metrics['value_loss']:.4f} | Elapsed: {elapsed:.1f}s"
            )

    vec_env.close()

    # Save checkpoint
    ckpt_path = f"ml/checkpoints/{arch_name}_bc_ppo_s{train_seed}.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "actor_hidden_dims": actor_hidden_dims,
            "seed": train_seed,
            "bc_r2": bc_r2,
            "eval_curves": eval_curves,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    return {
        "arch": arch_name,
        "seed": train_seed,
        "bc_r2": bc_r2,
        "eval_curves": eval_curves,
        "final_eval": eval_curves[-1],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [1001, 1002, 1003]
    total_steps = 100000

    all_results = []

    # 1. Train Teachers (16->64->32->1)
    for s in seeds:
        res = train_bc_ppo_run(
            arch_name="teacher",
            actor_hidden_dims=[64, 32],
            train_seed=s,
            total_steps=total_steps,
            eval_interval=25000,
            device=device,
        )
        all_results.append(res)

    # 2. Train Students (16->8->1)
    for s in seeds:
        res = train_bc_ppo_run(
            arch_name="student",
            actor_hidden_dims=[8],
            train_seed=s,
            total_steps=total_steps,
            eval_interval=25000,
            device=device,
        )
        all_results.append(res)

    with open("ml/checkpoints/bc_ppo_learning_curves.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved all learning curves to ml/checkpoints/bc_ppo_learning_curves.json")


if __name__ == "__main__":
    main()
