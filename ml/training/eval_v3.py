"""Evaluate V3 retrained models across seeds 50000..50029 and generate Canonical V3 Table."""

import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from ml.training.policy import ScorerPolicy
from userspace.trainer.env import SchedulerEnv
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from ml.training.evaluate import heuristic_select_action, classical_select_action


def eval_checkpoint_suite():
    eval_seeds = list(range(50000, 50030))
    scenarios = [
        ("pareto_0.5", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.5), 0.5),
        ("pareto_0.8", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8), 0.8),
        ("pareto_0.95", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.95), 0.95),
        ("poisson_0.5", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.5), 0.5),
        ("poisson_0.8", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.8), 0.8),
        ("poisson_0.95", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.95), 0.95),
        ("convoy_0.8", lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100), 0.8),
    ]

    # Load v3 models
    device = torch.device("cpu")

    def load_model(ckpt_path, dims):
        ckpt = torch.load(ckpt_path, map_location=device)
        p = ScorerPolicy(actor_hidden_dims=dims)
        st = {k[len("actor."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
        p.actor.load_state_dict(st)
        p.eval()
        return p

    teacher_v3_s1 = load_model("ml/checkpoints/ppo_v3_teacher_teacher_staged_s1001_checkpoint.pt", [64, 32])
    student_v3_s1 = load_model("ml/checkpoints/ppo_v3_student_student_staged_s1001_checkpoint.pt", [8])

    results = {}

    for sc_name, gen_func, rho in scenarios:
        print(f"\nEvaluating Scenario: {sc_name}...")
        results[sc_name] = {}

        # 1. Teacher v3
        mwts_t, p99s_t = [], []
        for s in eval_seeds:
            env = SchedulerEnv(workload_generator=gen_func, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                with torch.no_grad():
                    t_c = torch.from_numpy(obs["candidates"]).unsqueeze(0)
                    t_m = torch.from_numpy(obs["action_mask"]).unsqueeze(0)
                    act, _, _ = teacher_v3_s1.act(t_c, t_m, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(act.item()))
                done = term or trunc
            m = env._compute_episode_metrics()
            mwts_t.append(m["mean_waiting_time_us"])
            p99s_t.append(m["p99_waiting_time_us"])

        # 2. Student v3
        mwts_s, p99s_s = [], []
        for s in eval_seeds:
            env = SchedulerEnv(workload_generator=gen_func, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                with torch.no_grad():
                    t_c = torch.from_numpy(obs["candidates"]).unsqueeze(0)
                    t_m = torch.from_numpy(obs["action_mask"]).unsqueeze(0)
                    act, _, _ = student_v3_s1.act(t_c, t_m, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(act.item()))
                done = term or trunc
            m = env._compute_episode_metrics()
            mwts_s.append(m["mean_waiting_time_us"])
            p99s_s.append(m["p99_waiting_time_us"])

        # 3. Observation Heuristic (sigma=0.3 realistic target)
        mwts_h, p99s_h = [], []
        for s in eval_seeds:
            rng = np.random.default_rng(s)
            env = SchedulerEnv(workload_generator=gen_func, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs["action_mask"]
                valid = np.where(mask == 1)[0]
                cands = []
                if env.running_task is not None:
                    cands.append(env.running_task)
                cands.extend(env.ready_queue[: env.top_k - len(cands)])

                scores = []
                for i in valid:
                    t = cands[i]
                    noise = rng.normal(0.0, 0.30 * t.total_burst_us)
                    est = max(100.0, t.total_burst_us + noise)
                    scores.append(-(est / 100000.0) + 0.5 * (max(0, env.current_time_us - t.arrival_time_us - t.executed_burst_us) / 500000.0))

                act = valid[np.argmax(scores)]
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            m = env._compute_episode_metrics()
            mwts_h.append(m["mean_waiting_time_us"])
            p99s_h.append(m["p99_waiting_time_us"])

        # 4. Oracle Heuristic
        mwts_o, p99s_o = [], []
        for s in eval_seeds:
            env = SchedulerEnv(workload_generator=gen_func, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                cands = []
                if env.running_task is not None:
                    cands.append(env.running_task)
                cands.extend(env.ready_queue[: env.top_k - len(cands)])
                act = heuristic_select_action(cands, env.current_time_us)
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            m = env._compute_episode_metrics()
            mwts_o.append(m["mean_waiting_time_us"])
            p99s_o.append(m["p99_waiting_time_us"])

        def stat(arr):
            mean = np.mean(arr)
            ci = 1.96 * np.std(arr) / math.sqrt(len(arr))
            return round(float(mean), 1), round(float(ci), 1)

        results[sc_name] = {
            "Teacher-v3": {"mwt": stat(mwts_t), "p99": stat(p99s_t)},
            "Student-v3": {"mwt": stat(mwts_s), "p99": stat(p99s_s)},
            "Obs-Heuristic-sig0.3": {"mwt": stat(mwts_h), "p99": stat(p99s_h)},
            "Heuristic-Oracle": {"mwt": stat(mwts_o), "p99": stat(p99s_o)},
        }
        print(f"  Teacher-v3:           Mean WT = {stat(mwts_t)[0]:8.1f} ± {stat(mwts_t)[1]:5.1f} us | P99 = {stat(p99s_t)[0]:8.1f} us")
        print(f"  Student-v3:           Mean WT = {stat(mwts_s)[0]:8.1f} ± {stat(mwts_s)[1]:5.1f} us | P99 = {stat(p99s_s)[0]:8.1f} us")
        print(f"  Obs-Heuristic(sig=0.3):Mean WT = {stat(mwts_h)[0]:8.1f} ± {stat(mwts_h)[1]:5.1f} us | P99 = {stat(p99s_h)[0]:8.1f} us")
        print(f"  Heuristic-Oracle:     Mean WT = {stat(mwts_o)[0]:8.1f} ± {stat(mwts_o)[1]:5.1f} us | P99 = {stat(p99s_o)[0]:8.1f} us")

    with open("ml/checkpoints/canonical_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved canonical V3 evaluation to ml/checkpoints/canonical_v3_results.json")


if __name__ == "__main__":
    eval_checkpoint_suite()
