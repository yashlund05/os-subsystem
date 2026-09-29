import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn

from userspace.trainer.env import SchedulerEnv
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from ml.training.policy import CandidateScorer

def main():
    print("Collecting training data from SchedulerEnv rollouts on train seeds 1000..1029...")
    X_samples = []
    y_samples = []

    for s in range(1000, 1030):
        # Sample both Pareto and Multi-burst workloads for rich training diversity
        wl_gen = (
            (lambda seed: SyntheticWorkloadGenerator(seed=seed).generate_pareto_bursts(50, 1.3, 200, 0.8))
            if s % 2 == 0 else
            (lambda seed: SyntheticWorkloadGenerator(seed=seed).generate_multiburst_process_workload(10, 10, 1.3, 200))
        )
        env = SchedulerEnv(workload_generator=wl_gen, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs['action_mask']
            valid = np.where(mask == 1)[0]
            for idx in valid:
                feat = obs['candidates'][idx]
                X_samples.append(feat)
                p_cue = max(0.0, float(feat[15] - feat[9]))
                y_samples.append(-feat[1] + 1.0 * feat[2] + 2.0 * p_cue)
            scores = [
                -obs['candidates'][i, 1] + 1.0 * obs['candidates'][i, 2] + 2.0 * max(0.0, float(obs['candidates'][i, 15] - obs['candidates'][i, 9]))
                for i in valid
            ]
            act = valid[np.argmax(scores)]
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc

    X = np.array(X_samples)
    y = np.array(y_samples)

    # 1. Closed-form OLS check
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_ols = X @ w
    r2_ols = 1.0 - np.sum((y - y_ols)**2) / np.sum((y - np.mean(y))**2)
    print(f"Closed-form OLS R2: {r2_ols:.6f}")
    print(f"Weights on feat[1] (burst_est), feat[2] (age), feat[9] (preempt_cue): {w[1]:.4f}, {w[2]:.4f}, {w[9]:.4f}")
    other_indices = [i for i in range(16) if i not in [1, 2, 9]]
    print(f"Max abs weight on all other 13 features: {np.max(np.abs(w[other_indices])):.2e}")

    # 2. Standardized Neural Net 16->8->1 training
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    torch.manual_seed(42)
    student = CandidateScorer(hidden_dims=[8])
    opt = torch.optim.Adam(student.parameters(), lr=0.01)
    crit = nn.MSELoss()

    for ep in range(1000):
        pred = student(X_t).squeeze(-1)
        loss = crit(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pred_raw = pred.detach().numpy() * y_std + y_mean
    r2_nn = 1.0 - np.sum((y - pred_raw)**2) / np.sum((y - np.mean(y))**2)
    print(f"Standardized Target Neural Net (16->8->1) R2: {r2_nn:.6f}, final loss: {loss.item():.6f}")

    # 3. Evaluate across Pareto, Poisson, Convoy, and Multi-burst
    workloads = [
        ("Pareto rho=0.8", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8)),
        ("Poisson rho=0.8", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.8)),
        ("Convoy", lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100)),
        ("Multi-Burst", lambda s: SyntheticWorkloadGenerator(seed=s).generate_multiburst_process_workload(10, 10, 1.3, 200)),
    ]

    print(f"\n{'Workload':<18} | {'Top-1 Agree':<18} | {'Student Mean WT':<22} | {'Heuristic Mean WT':<22}")
    print("-" * 86)

    for name, gen_fn in workloads:
        matches = 0
        total_dec = 0
        mwts_s = []
        mwts_h = []

        for s in range(50000, 50030):
            # Heuristic rollout & agreement check
            env = SchedulerEnv(workload_generator=gen_fn, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs['action_mask']
                valid = np.where(mask == 1)[0]
                if len(valid) > 1:
                    total_dec += 1
                    scores_h = [
                        -obs['candidates'][i, 1] + 1.0 * obs['candidates'][i, 2] + 2.0 * max(0.0, float(obs['candidates'][i, 15] - obs['candidates'][i, 9]))
                        for i in valid
                    ]
                    act_h = valid[np.argmax(scores_h)]
                    with torch.no_grad():
                        c_t = torch.from_numpy(obs['candidates']).float()
                        s_out = student(c_t).squeeze(-1).numpy()
                    s_out[mask == 0] = -1e9
                    act_s = int(np.argmax(s_out))
                    if act_h == act_s:
                        matches += 1
                scores = [
                    -obs['candidates'][i, 1] + 1.0 * obs['candidates'][i, 2] + 2.0 * max(0.0, float(obs['candidates'][i, 15] - obs['candidates'][i, 9]))
                    for i in valid
                ]
                act = valid[np.argmax(scores)]
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            mwts_h.append(env._compute_episode_metrics()['mean_waiting_time_us'])

            # Student rollout
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs['action_mask']
                with torch.no_grad():
                    c_t = torch.from_numpy(obs['candidates']).float()
                    s_out = student(c_t).squeeze(-1).numpy()
                s_out[mask == 0] = -1e9
                act_s = int(np.argmax(s_out))
                obs, _, term, trunc, _ = env.step(act_s)
                done = term or trunc
            mwts_s.append(env._compute_episode_metrics()['mean_waiting_time_us'])

        pct = 100.0 * matches / max(1, total_dec)
        m_s, ci_s = np.mean(mwts_s), 1.96 * np.std(mwts_s) / math.sqrt(len(mwts_s))
        m_h, ci_h = np.mean(mwts_h), 1.96 * np.std(mwts_h) / math.sqrt(len(mwts_h))
        print(f"{name:<18} | {pct:6.2f}% ({matches}/{total_dec}) | {m_s:7.1f} ± {ci_s:5.1f} us     | {m_h:7.1f} ± {ci_h:5.1f} us")

    # Save model checkpoint
    torch.save({
        "student_state_dict": student.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std,
    }, "ml/checkpoints/supervised_student_v4.pt")
    print("\nSaved student checkpoint to ml/checkpoints/supervised_student_v4.pt")

if __name__ == "__main__":
    main()
