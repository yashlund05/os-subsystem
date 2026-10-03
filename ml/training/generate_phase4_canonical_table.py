"""Canonical Phase 4 Benchmark Generator.

Evaluates Quantized-Student (int8), Float-Student (FP32), Heuristic-Obs, and Supervised-Student
across all 8 canonical workloads and 30 disjoint evaluation seeds (50000..50029).
Computes Mean WT (with 95% CI), P99, Max Wait, Context Switches, and Degradation %.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import List

import numpy as np
import torch

from ml.training.policy import CandidateScorer, ScorerPolicy
from quantization.int8_forward import int8_forward_batch
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv


def ci95(arr: List[float]) -> float:
    n = len(arr)
    if n <= 1:
        return 0.0
    s = float(np.std(arr, ddof=1))
    t_val = 2.045 if n == 30 else 1.96
    return t_val * (s / math.sqrt(n))


def fmt_stat(arr: List[float]) -> str:
    m = float(np.mean(arr))
    if len(arr) <= 1:
        return f"{m:7.1f} ±   0.0"
    c = ci95(arr)
    return f"{m:7.1f} ± {c:5.1f}"


def run_benchmark():
    eval_seeds = list(range(50000, 50030))
    print(f"Running Phase 4 Benchmark across {len(eval_seeds)} eval seeds (50000..50029)...")

    # 1. Load Float Student
    float_ckpt = torch.load(
        "ml/checkpoints/student_bc_ppo_s1003.pt", map_location="cpu", weights_only=False
    )
    float_student = ScorerPolicy(
        actor_hidden_dims=float_ckpt["actor_hidden_dims"], critic_hidden_dims=[64, 64]
    )
    float_student.load_state_dict(float_ckpt["model_state_dict"])
    float_student.eval()

    # 2. Load Quantized Student
    qdata = np.load("ml/checkpoints/quantized_student_int8.npz")
    with open("ml/checkpoints/quantized_student_int8.json", "r") as f:
        qmeta = json.load(f)
    s_x = np.array(qmeta["scales"]["s_x"], dtype=np.float64)
    w1 = qdata["w1"]
    b1 = qdata["b1"]
    w2 = qdata["w2"]
    b2 = qdata["b2"]

    # 3. Load Supervised Student
    sup_ckpt = torch.load(
        "ml/checkpoints/supervised_student_v4.pt", map_location="cpu", weights_only=False
    )
    sup_student = CandidateScorer(hidden_dims=[8])
    sup_student.load_state_dict(sup_ckpt["student_state_dict"])
    sup_student.eval()

    workloads = {
        "Pareto rho=0.5": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.3, 200, 0.5
        ),
        "Pareto rho=0.8": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.3, 200, 0.8
        ),
        "Pareto rho=0.95": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.3, 200, 0.95
        ),
        "Poisson rho=0.5": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.8, 200, 0.5
        ),
        "Poisson rho=0.8": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.8, 200, 0.8
        ),
        "Poisson rho=0.95": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(
            50, 1.8, 200, 0.95
        ),
        "Convoy": lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100),
        "Multi-Burst (OOD Test)": lambda s: SyntheticWorkloadGenerator(
            seed=s
        ).generate_multiburst_process_workload(10, 10, 1.3, 200),
    }

    results = {}

    for wl_name, wl_fn in workloads.items():
        print(f"Evaluating {wl_name}...")
        row = {
            "Float-Student": {"mean_wt": [], "p99_wt": [], "max_wait": [], "switches": []},
            "Quantized-Student": {"mean_wt": [], "p99_wt": [], "max_wait": [], "switches": []},
            "Heuristic-Obs": {"mean_wt": [], "p99_wt": [], "max_wait": [], "switches": []},
            "Supervised-Student": {"mean_wt": [], "p99_wt": [], "max_wait": [], "switches": []},
        }

        # Seeds loop
        seeds_to_run = [50000] if wl_name == "Convoy" else eval_seeds

        for s in seeds_to_run:
            # A. Float-Student
            env = SchedulerEnv(workload_generator=wl_fn, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs["action_mask"]
                cands = obs["candidates"]
                with torch.no_grad():
                    c_t = torch.from_numpy(cands).unsqueeze(0)
                    m_t = torch.from_numpy(mask).unsqueeze(0)
                    act, _, _ = float_student.act(c_t, m_t, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(act.item()))
                done = term or trunc
            m = env._compute_episode_metrics()
            row["Float-Student"]["mean_wt"].append(m["mean_waiting_time_us"])
            row["Float-Student"]["p99_wt"].append(m["p99_waiting_time_us"])
            row["Float-Student"]["max_wait"].append(
                max(t.waiting_time_us for t in env.completed_tasks)
            )
            row["Float-Student"]["switches"].append(float(env.total_context_switches))

            # B. Quantized-Student (Pure int8 forward)
            env = SchedulerEnv(workload_generator=wl_fn, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs["action_mask"]
                cands = obs["candidates"]
                q_feats = np.clip(np.round(cands / s_x.reshape(1, 16)), -128, 127).astype(np.int8)
                int_scores = int8_forward_batch(q_feats, w1, b1, w2, b2)
                int_scores[mask == 0] = -2147483648
                act_int = int(np.argmax(int_scores))
                obs, _, term, trunc, _ = env.step(act_int)
                done = term or trunc
            m = env._compute_episode_metrics()
            row["Quantized-Student"]["mean_wt"].append(m["mean_waiting_time_us"])
            row["Quantized-Student"]["p99_wt"].append(m["p99_waiting_time_us"])
            row["Quantized-Student"]["max_wait"].append(
                max(t.waiting_time_us for t in env.completed_tasks)
            )
            row["Quantized-Student"]["switches"].append(float(env.total_context_switches))

            # C. Heuristic-Obs (Real predictor)
            env = SchedulerEnv(workload_generator=wl_fn, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs["action_mask"]
                valid = np.where(mask == 1)[0]
                scores = [
                    -obs["candidates"][i, 1]
                    + 1.0 * obs["candidates"][i, 2]
                    + 2.0 * obs["candidates"][i, 9]
                    for i in valid
                ]
                act = valid[np.argmax(scores)]
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            m = env._compute_episode_metrics()
            row["Heuristic-Obs"]["mean_wt"].append(m["mean_waiting_time_us"])
            row["Heuristic-Obs"]["p99_wt"].append(m["p99_waiting_time_us"])
            row["Heuristic-Obs"]["max_wait"].append(
                max(t.waiting_time_us for t in env.completed_tasks)
            )
            row["Heuristic-Obs"]["switches"].append(float(env.total_context_switches))

            # D. Supervised-Student
            env = SchedulerEnv(workload_generator=wl_fn, top_k=16)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs["action_mask"]
                with torch.no_grad():
                    c_t = torch.from_numpy(obs["candidates"]).float()
                    s_out = sup_student(c_t).squeeze(-1).numpy()
                s_out[mask == 0] = -1e9
                act = int(np.argmax(s_out))
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            m = env._compute_episode_metrics()
            row["Supervised-Student"]["mean_wt"].append(m["mean_waiting_time_us"])
            row["Supervised-Student"]["p99_wt"].append(m["p99_waiting_time_us"])
            row["Supervised-Student"]["max_wait"].append(
                max(t.waiting_time_us for t in env.completed_tasks)
            )
            row["Supervised-Student"]["switches"].append(float(env.total_context_switches))

        results[wl_name] = row

    # Build canonical markdown table
    print("\n" + "=" * 110)
    print("PHASE 4 CANONICAL BENCHMARK: QUANTIZED STUDENT VS FLOAT STUDENT (Seeds 50000..50029)")
    print("=" * 110)

    header = "| Workload | Float-Student | Quantized-Student (int8) | Degradation (%) | Heuristic-Obs | Supervised-Student | Status |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    print(header)
    print(sep)

    table_rows = [header, sep]

    for wl_name, row in results.items():
        m_fl = float(np.mean(row["Float-Student"]["mean_wt"]))
        m_q = float(np.mean(row["Quantized-Student"]["mean_wt"]))
        deg = ((m_q - m_fl) / m_fl) * 100.0 if m_fl > 0 else 0.0

        # Status threshold
        limit = 10.0 if "OOD" in wl_name else 5.0
        status = "PASSED" if deg <= limit else f"FLAGGED (> {limit}%)"

        line = (
            f"| **{wl_name}** "
            f"| {fmt_stat(row['Float-Student']['mean_wt'])} "
            f"| {fmt_stat(row['Quantized-Student']['mean_wt'])} "
            f"| {deg:+6.2f}% "
            f"| {fmt_stat(row['Heuristic-Obs']['mean_wt'])} "
            f"| {fmt_stat(row['Supervised-Student']['mean_wt'])} "
            f"| **{status}** |"
        )
        print(line)
        table_rows.append(line)

    # Save to file
    out_table_path = Path("ml/checkpoints/canonical_phase4_table.md")
    with open(out_table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_rows) + "\n")

    out_json_path = Path("ml/checkpoints/canonical_phase4_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        # Convert numpy floats for serialization
        clean_res = {
            wl: {
                pol: {k: [float(x) for x in v] for k, v in pdata.items()}
                for pol, pdata in wdata.items()
            }
            for wl, wdata in results.items()
        }
        json.dump(clean_res, f, indent=2)

    print(f"\nSaved benchmark results to {out_table_path} and {out_json_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
