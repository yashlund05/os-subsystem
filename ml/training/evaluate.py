"""Comprehensive 30+ seed evaluation harness for NeuroOS scheduling policies.

Compares:
- Learned PPO Policies (Teacher 16->64->32->1 and Student 16->8->1)
- Classical baselines: FCFS, SJF, SRTF, Round Robin (5ms), MLFQ
- Heuristic baseline on same observation: argmin(predicted_burst) with age tie-break/bonus
- Noise sensitivity analysis: Policy performance at sigma = 0, default, and high noise
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ml.training.policy import ScorerPolicy
from simulator.scheduling.task import SimulatedTask
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv


def compute_student_t_ci(
    values: Sequence[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Returns (mean, std, 95% half-width margin of error)."""
    n = len(values)
    if n < 2:
        return float(values[0]) if n == 1 else 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    # Approximate t-critical for n >= 30 (1.96 - 2.04)
    t_crit = 2.042 if n < 30 else 1.96
    margin = float(t_crit * (std / math.sqrt(n)))
    return mean, std, margin


def heuristic_select_action(candidates: List[SimulatedTask], current_time_us: int) -> int:
    """
    Observation-space heuristic baseline:
    Ranks candidates by argmin(predicted_burst - 0.2 * age).
    """
    if not candidates:
        return 0
    best_idx = 0
    best_score = 1e12
    for i, t in enumerate(candidates):
        age = max(0, current_time_us - t.arrival_time_us - t.executed_burst_us)
        # Using remaining/executed observable attributes
        pred = max(100.0, float(t.total_burst_us))  # or estimate
        score = pred - 0.2 * age
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


def classical_select_action(
    policy_name: str, candidates: List[SimulatedTask], current_time_us: int
) -> int:
    """Dispatches action for classical scheduling algorithms."""
    if not candidates:
        return 0
    if policy_name == "FCFS":
        best_idx = 0
        best_arr = candidates[0].arrival_time_us
        for i, t in enumerate(candidates):
            if t.arrival_time_us < best_arr:
                best_arr = t.arrival_time_us
                best_idx = i
        return best_idx
    elif policy_name == "SJF":
        best_idx = 0
        best_b = candidates[0].total_burst_us
        for i, t in enumerate(candidates):
            if t.total_burst_us < best_b:
                best_b = t.total_burst_us
                best_idx = i
        return best_idx
    elif policy_name == "SRTF":
        best_idx = 0
        best_r = candidates[0].remaining_burst_us
        for i, t in enumerate(candidates):
            if t.remaining_burst_us < best_r:
                best_r = t.remaining_burst_us
                best_idx = i
        return best_idx
    elif policy_name == "MLFQ":
        # Multi-Level Feedback Queue: pick task with lowest priority_level (0 is highest)
        best_idx = 0
        best_lvl = getattr(candidates[0], "priority_level", 0)
        for i, t in enumerate(candidates):
            lvl = getattr(t, "priority_level", 0)
            if lvl < best_lvl:
                best_lvl = lvl
                best_idx = i
        return best_idx
    elif policy_name == "RR-5ms":
        return 0
    return 0


def run_evaluation_suite(
    checkpoint_path: Optional[str] = None,
    num_eval_seeds: int = 30,
    eval_seed_base: int = 50000,
    loads: Optional[List[float]] = None,
    workload_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs paired evaluations across all policies and configurations."""
    if loads is None:
        loads = [0.50, 0.80, 0.95]
    if workload_types is None:
        workload_types = ["pareto", "poisson", "convoy"]

    device = torch.device("cpu")
    policy_model: Optional[ScorerPolicy] = None

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        actor_dims = ckpt.get("actor_hidden_dims", [64, 32])
        policy_model = ScorerPolicy(actor_hidden_dims=actor_dims)
        # Extract and load only actor state dict (critic is not needed for evaluation)
        actor_state = {
            k[len("actor.") :]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("actor.")
        }
        policy_model.actor.load_state_dict(actor_state)
        policy_model.eval()

    eval_seeds = [eval_seed_base + i for i in range(num_eval_seeds)]

    policies_to_test = ["FCFS", "SJF", "SRTF", "RR-5ms", "MLFQ", "Heuristic"]
    if policy_model is not None:
        policies_to_test.extend(["PPO-Policy", "PPO-Noise0", "PPO-NoiseHigh"])

    results: Dict[str, Any] = {}

    for w_type in workload_types:
        results[w_type] = {}
        for rho in loads:
            case_key = f"load_{rho}"
            results[w_type][case_key] = {}

            print(
                f"\n--- Evaluating Workload: {w_type.upper()} | Load rho={rho} (Seeds {eval_seeds[0]}..{eval_seeds[-1]}) ---"
            )

            for pol_name in policies_to_test:
                mwts, p99s, max_wts, ctxs = [], [], [], []

                for s in eval_seeds:
                    # Configure noise level for sensitivity analysis
                    noise_sigma = 0.20
                    if pol_name == "PPO-Noise0":
                        noise_sigma = 0.0
                    elif pol_name == "PPO-NoiseHigh":
                        noise_sigma = 0.60

                    def gen(
                        seed: int, _w_type: str = w_type, _rho: float = rho
                    ) -> List[SimulatedTask]:
                        if _w_type == "convoy":
                            return AdversarialWorkloadGenerator.create_convoy_workload(
                                num_short_jobs=49, long_burst_us=50000, short_burst_us=100
                            )
                        else:
                            synth = SyntheticWorkloadGenerator(seed=seed)
                            return synth.generate_pareto_bursts(
                                num_tasks=50,
                                alpha=1.3 if _w_type == "pareto" else 1.8,
                                min_burst_us=200,
                                load_factor=_rho,
                            )

                    env = SchedulerEnv(
                        workload_generator=gen,
                        load_factor=rho,
                        max_steps=1000,
                        top_k=16,
                    )
                    env.burst_estimator.noise_std_frac = noise_sigma

                    obs, info = env.reset(seed=s)
                    done = False
                    last_info = info

                    while not done:
                        mask = obs["action_mask"]
                        valid = np.where(mask == 1)[0]
                        if len(valid) == 0:
                            action = 0
                        elif pol_name.startswith("PPO-") and policy_model is not None:
                            t_cands = torch.from_numpy(obs["candidates"]).unsqueeze(0)
                            t_mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0)
                            with torch.no_grad():
                                act, _, _ = policy_model.act(t_cands, t_mask, deterministic=True)
                                action = int(act.item())
                        else:
                            # Candidate list
                            candidates: List[SimulatedTask] = []
                            if env.running_task is not None:
                                candidates.append(env.running_task)
                            candidates.extend(env.ready_queue[: env.top_k - len(candidates)])

                            if pol_name == "Heuristic":
                                action = heuristic_select_action(candidates, env.current_time_us)
                            else:
                                action = classical_select_action(
                                    pol_name, candidates, env.current_time_us
                                )

                        obs, r, term, trunc, info = env.step(action)
                        done = term or trunc
                        last_info = info

                    metrics = last_info.get("metrics", env._compute_episode_metrics())
                    mwts.append(metrics["mean_waiting_time_us"])
                    p99s.append(metrics["p99_waiting_time_us"])
                    max_wts.append(max([t.waiting_time_us for t in env.completed_tasks] or [0]))
                    ctxs.append(metrics["total_context_switches"])

                m_mwt, s_mwt, ci_mwt = compute_student_t_ci(mwts)
                m_p99, s_p99, ci_p99 = compute_student_t_ci(p99s)
                m_max, s_max, ci_max = compute_student_t_ci(max_wts)
                m_ctx, _, _ = compute_student_t_ci(ctxs)

                results[w_type][case_key][pol_name] = {
                    "mean_wt_us": round(m_mwt, 2),
                    "mean_wt_ci": round(ci_mwt, 2),
                    "p99_wt_us": round(m_p99, 2),
                    "p99_wt_ci": round(ci_p99, 2),
                    "max_wt_us": round(m_max, 2),
                    "context_switches": round(m_ctx, 1),
                }

                print(
                    f"  {pol_name:<14} | Mean WT: {m_mwt:8.1f} ± {ci_mwt:5.1f} us | "
                    f"P99: {m_p99:8.1f} ± {ci_p99:5.1f} us | Max: {m_max:8.1f} us | "
                    f"Ctx: {m_ctx:5.1f}"
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuroOS Policies vs Baselines")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint path (.pt)")
    parser.add_argument("--seeds", type=int, default=30, help="Number of eval seeds (>=30)")
    parser.add_argument(
        "--output", type=str, default="ml/checkpoints/eval_results.json", help="Output JSON"
    )
    args = parser.parse_args()

    results = run_evaluation_suite(
        checkpoint_path=args.checkpoint,
        num_eval_seeds=args.seeds,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive evaluation table to {out_p}")


if __name__ == "__main__":
    main()
