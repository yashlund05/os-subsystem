"""Hyperparameter sensitivity tuning for Teacher architecture to determine if the gap with Student closes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
from typing import Dict

import numpy as np
import torch

from ml.training.policy import ScorerPolicy
from ml.training.ppo import PPOConfig, PPOTrainer
from ml.training.train_bc_ppo import pretrain_actor_bc
from ml.training.vec_env import VectorSchedulerEnv
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv
from userspace.trainer.reward import RewardConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_seeds = list(range(50000, 50030))


def eval_30(policy: ScorerPolicy) -> Dict[str, float]:
    policy.eval()
    pareto_wts, poisson_wts = [], []
    def wl_pareto(s):
        return SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8)
    def wl_poisson(s):
        return SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.8)

    for s in eval_seeds:
        for wl, arr in [(wl_pareto, pareto_wts), (wl_poisson, poisson_wts)]:
            env = SchedulerEnv(workload_generator=wl, top_k=16)
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
            m = env._compute_episode_metrics()
            arr.append(m["mean_waiting_time_us"])
    return {"pareto_08": float(np.mean(pareto_wts)), "poisson_08": float(np.mean(poisson_wts))}


def main():
    configs = [
        {
            "name": "Config-A (Conservative: lr=3e-5, ent=0.005->0.0005, clip=0.1)",
            "cfg": PPOConfig(
                learning_rate=0.00003,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.1,
                value_coef=0.5,
                entropy_coef_start=0.005,
                entropy_coef_end=0.0005,
                max_grad_norm=0.5,
                num_rollout_steps=128,
                num_epochs=4,
                batch_size=128,
                target_kl=0.015,
            ),
        },
        {
            "name": "Config-B (High Exploration: lr=5e-5, ent=0.03->0.005, clip=0.2)",
            "cfg": PPOConfig(
                learning_rate=0.00005,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                value_coef=0.5,
                entropy_coef_start=0.03,
                entropy_coef_end=0.005,
                max_grad_norm=0.5,
                num_rollout_steps=128,
                num_epochs=4,
                batch_size=128,
                target_kl=0.03,
            ),
        },
    ]

    rew_cfg = RewardConfig(w_wait=1.0, w_completion=0.0, w_switch=0.02, w_starvation=0.1, w_tail_threshold=0.1)

    results = []

    for c in configs:
        print("=" * 70)
        print("Testing", c["name"])
        print("=" * 70)
        torch.manual_seed(1001)
        np.random.seed(1001)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(1001)

        teacher = ScorerPolicy(actor_hidden_dims=[64, 32], critic_hidden_dims=[64, 64]).to(device)
        pretrain_actor_bc(teacher, list(range(1001, 1011)), device, epochs=300)

        e0 = eval_30(teacher)
        print(f"  Step 0: Pareto 0.8 = {e0['pareto_08']:.1f} us, Poisson 0.8 = {e0['poisson_08']:.1f} us")

        env_fns = [
            (
                lambda idx=i: SchedulerEnv(
                    workload_generator=(
                        lambda seed: SyntheticWorkloadGenerator(
                            seed=(1001 * 1000 + idx * 100 + seed) % 1000000
                        ).generate_pareto_bursts(50, 1.3, 200, 0.8)
                    ),
                    top_k=16,
                    reward_config=rew_cfg,
                )
            )
            for i in range(8)
        ]
        vec_env = VectorSchedulerEnv(env_fns)
        trainer = PPOTrainer(policy=teacher, vec_env=vec_env, config=c["cfg"], device=device)

        obs_cands, obs_masks = vec_env.reset()
        t0 = time.time()
        while trainer.total_timesteps < 50000:
            progress = float(trainer.total_timesteps) / 50000.0
            buffers, obs_cands, obs_masks, _ = trainer.collect_rollouts(obs_cands, obs_masks)
            trainer.update(buffers, progress=progress)

        ef = eval_30(teacher)
        elapsed = time.time() - t0
        print(f"  Step 50k ({elapsed:.1f}s): Pareto 0.8 = {ef['pareto_08']:.1f} us, Poisson 0.8 = {ef['poisson_08']:.1f} us")
        results.append({"name": c["name"], "step0": e0, "step50k": ef})

    print("\n" + "=" * 70)
    print("SUMMARY OF TEACHER RETUNING RESULTS")
    print("=" * 70)
    for r in results:
        print(f"{r['name']}:")
        print(f"  Pareto rho=0.8: {r['step0']['pareto_08']:.1f} us -> {r['step50k']['pareto_08']:.1f} us")
        print(f"  Poisson rho=0.8: {r['step0']['poisson_08']:.1f} us -> {r['step50k']['poisson_08']:.1f} us")


if __name__ == "__main__":
    main()
