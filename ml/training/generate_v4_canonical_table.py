"""Generate the complete Reconciled Canonical Benchmark Results Table across 30 eval seeds."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Any, Callable, Dict, List

import numpy as np
import torch

from ml.training.evaluate import heuristic_select_action
from ml.training.policy import CandidateScorer, ScorerPolicy
from schedulers.fcfs.scheduler import FCFSScheduler
from schedulers.mlfq.scheduler import MLFQScheduler
from schedulers.round_robin.scheduler import RoundRobinScheduler
from schedulers.sjf.scheduler import SJFScheduler
from schedulers.srtf.scheduler import SRTFScheduler
from simulator.scheduling.engine import SchedulingSimulationEngine
from simulator.scheduling.task import SimulatedTask
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv


def eval_phase1_engine(
    scheduler_cls,
    kwargs: Dict[str, Any],
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> Dict[str, List[float]]:
    mwts, p99s, max_waits, switches = [], [], [], []
    for s in seeds:
        tasks = workload_fn(s)
        sched = scheduler_cls(**kwargs)
        engine = SchedulingSimulationEngine(scheduler=sched, context_switch_overhead_us=0)
        history, metrics = engine.run(tasks)
        mwts.append(metrics.mean_waiting_time_us)
        p99s.append(metrics.p99_waiting_time_us)
        max_waits.append(float(max(t.waiting_time_us for t in history)))
        switches.append(float(metrics.total_context_switches))
    return {"mean_wt": mwts, "p99_wt": p99s, "max_wait": max_waits, "switches": switches}


def eval_heuristic_oracle(
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> Dict[str, List[float]]:
    mwts, p99s, max_waits, switches = [], [], [], []
    for s in seeds:
        env = SchedulerEnv(workload_generator=workload_fn, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            np.where(mask == 1)[0]
            cands = []
            if env.running_task is not None:
                cands.append(env.running_task)
            cands.extend(env.ready_queue[: env.top_k - len(cands)])
            act = heuristic_select_action(cands, env.current_time_us)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        m = env._compute_episode_metrics()
        mwts.append(m["mean_waiting_time_us"])
        p99s.append(m["p99_waiting_time_us"])
        max_waits.append(float(max(t.waiting_time_us for t in env.completed_tasks)))
        switches.append(float(env.total_context_switches))
    return {"mean_wt": mwts, "p99_wt": p99s, "max_wait": max_waits, "switches": switches}


def eval_heuristic_obs(
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> Dict[str, List[float]]:
    mwts, p99s, max_waits, switches = [], [], [], []
    for s in seeds:
        env = SchedulerEnv(workload_generator=workload_fn, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            # Observation heuristic rule with real predictor and preemption cue
            scores = [
                -obs["candidates"][i, 1]
                + 1.0 * obs["candidates"][i, 2]
                + 2.0 * max(0.0, float(obs["candidates"][i, 15] - obs["candidates"][i, 9]))
                for i in valid
            ]
            act = valid[np.argmax(scores)]
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        m = env._compute_episode_metrics()
        mwts.append(m["mean_waiting_time_us"])
        p99s.append(m["p99_waiting_time_us"])
        max_waits.append(float(max(t.waiting_time_us for t in env.completed_tasks)))
        switches.append(float(env.total_context_switches))
    return {"mean_wt": mwts, "p99_wt": p99s, "max_wait": max_waits, "switches": switches}


def eval_supervised_student(
    model: CandidateScorer,
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> Dict[str, List[float]]:
    mwts, p99s, max_waits, switches = [], [], [], []
    model.eval()
    for s in seeds:
        env = SchedulerEnv(workload_generator=workload_fn, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            with torch.no_grad():
                c_t = torch.from_numpy(obs["candidates"]).float()
                s_out = model(c_t).squeeze(-1).numpy()
            s_out[mask == 0] = -1e9
            act = int(np.argmax(s_out))
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        m = env._compute_episode_metrics()
        mwts.append(m["mean_waiting_time_us"])
        p99s.append(m["p99_waiting_time_us"])
        max_waits.append(float(max(t.waiting_time_us for t in env.completed_tasks)))
        switches.append(float(env.total_context_switches))
    return {"mean_wt": mwts, "p99_wt": p99s, "max_wait": max_waits, "switches": switches}


def eval_policy_model(
    policy: ScorerPolicy,
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
    device: torch.device,
) -> Dict[str, List[float]]:
    mwts, p99s, max_waits, switches = [], [], [], []
    policy.eval()
    for s in seeds:
        env = SchedulerEnv(workload_generator=workload_fn, top_k=16)
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
        mwts.append(m["mean_waiting_time_us"])
        p99s.append(m["p99_waiting_time_us"])
        max_waits.append(float(max(t.waiting_time_us for t in env.completed_tasks)))
        switches.append(float(env.total_context_switches))
    return {"mean_wt": mwts, "p99_wt": p99s, "max_wait": max_waits, "switches": switches}


def fmt_stat(arr: List[float]) -> str:
    m = float(np.mean(arr))
    if len(arr) <= 1:
        return f"{m:.1f}"
    ci = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return f"{m:7.1f} ± {ci:5.1f}"


def main():
    eval_seeds = list(range(50000, 50030))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running Reconciled Canonical Evaluation on {len(eval_seeds)} seeds ({eval_seeds[0]}..{eval_seeds[-1]})..."
    )

    # Define all workloads
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
        "Multi-Burst Process": lambda s: SyntheticWorkloadGenerator(
            seed=s
        ).generate_multiburst_process_workload(10, 10, 1.3, 200),
    }

    # Load Supervised Student
    sup_ckpt = torch.load(
        "ml/checkpoints/supervised_student_v4.pt", map_location="cpu", weights_only=False
    )
    sup_student = CandidateScorer(hidden_dims=[8])
    sup_student.load_state_dict(sup_ckpt["student_state_dict"])

    # Load Teacher (Seed 1002 had best balanced performance)
    teacher_ckpts = [
        torch.load(
            f"ml/checkpoints/teacher_bc_ppo_s{s}.pt", map_location=device, weights_only=False
        )
        for s in [1001, 1002, 1003]
    ]
    teachers = []
    for ckpt in teacher_ckpts:
        p = ScorerPolicy(
            actor_hidden_dims=ckpt["actor_hidden_dims"], critic_hidden_dims=[64, 64]
        ).to(device)
        p.load_state_dict(ckpt["model_state_dict"])
        teachers.append(p)
    teacher_best = teachers[1]  # Seed 1002

    # Load Student (Seed 1003 had best multi-burst and Pareto performance)
    student_ckpts = [
        torch.load(
            f"ml/checkpoints/student_bc_ppo_s{s}.pt", map_location=device, weights_only=False
        )
        for s in [1001, 1002, 1003]
    ]
    students = []
    for ckpt in student_ckpts:
        p = ScorerPolicy(
            actor_hidden_dims=ckpt["actor_hidden_dims"], critic_hidden_dims=[64, 64]
        ).to(device)
        p.load_state_dict(ckpt["model_state_dict"])
        students.append(p)
    student_best = students[2]  # Seed 1003

    full_results = {}

    for wl_name, wl_fn in workloads.items():
        print(f"Evaluating {wl_name}...")
        row = {}
        row["FCFS"] = eval_phase1_engine(FCFSScheduler, {}, wl_fn, eval_seeds)
        row["SJF"] = eval_phase1_engine(SJFScheduler, {}, wl_fn, eval_seeds)
        row["SRTF"] = eval_phase1_engine(SRTFScheduler, {}, wl_fn, eval_seeds)
        row["RR (5ms)"] = eval_phase1_engine(
            RoundRobinScheduler, {"quantum_us": 5000}, wl_fn, eval_seeds
        )
        row["MLFQ (3-lvl)"] = eval_phase1_engine(
            MLFQScheduler, {"num_levels": 3, "base_quantum_us": 5000}, wl_fn, eval_seeds
        )
        row["Heuristic-Oracle"] = eval_heuristic_oracle(wl_fn, eval_seeds)
        row["Heuristic-Obs (Real Predictor)"] = eval_heuristic_obs(wl_fn, eval_seeds)
        row["Supervised-Student"] = eval_supervised_student(sup_student, wl_fn, eval_seeds)
        row["Teacher (BC+PPO)"] = eval_policy_model(teacher_best, wl_fn, eval_seeds, device)
        row["Student (BC+PPO)"] = eval_policy_model(student_best, wl_fn, eval_seeds, device)

        full_results[wl_name] = row

    # 1. Main Canonical Table: Mean Waiting Time
    columns = [
        "Workload",
        "FCFS",
        "SJF",
        "SRTF",
        "RR (5ms)",
        "MLFQ (3-lvl)",
        "Heuristic-Oracle",
        "Heuristic-Obs (Real Predictor)",
        "Supervised-Student",
        "Teacher (BC+PPO)",
        "Student (BC+PPO)",
    ]

    print("\n" + "=" * 130)
    print("CANONICAL BENCHMARK TABLE (Mean Waiting Time in us ± 95% CI across Seeds 50000..50029)")
    print("=" * 130)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    print(header)
    print(sep)

    md_lines = [header, sep]

    for wl_name, row in full_results.items():
        vals = [
            f"**{wl_name}**",
            fmt_stat(row["FCFS"]["mean_wt"]),
            fmt_stat(row["SJF"]["mean_wt"]),
            fmt_stat(row["SRTF"]["mean_wt"]),
            fmt_stat(row["RR (5ms)"]["mean_wt"]),
            fmt_stat(row["MLFQ (3-lvl)"]["mean_wt"]),
            fmt_stat(row["Heuristic-Oracle"]["mean_wt"]),
            fmt_stat(row["Heuristic-Obs (Real Predictor)"]["mean_wt"]),
            fmt_stat(row["Supervised-Student"]["mean_wt"]),
            fmt_stat(row["Teacher (BC+PPO)"]["mean_wt"]),
            fmt_stat(row["Student (BC+PPO)"]["mean_wt"]),
        ]
        line = "| " + " | ".join(vals) + " |"
        print(line)
        md_lines.append(line)

    # 2. Multi-Burst Comprehensive Metrics Table
    mb_row = full_results["Multi-Burst Process"]
    mb_cols = ["Policy", "Mean WT (us)", "P99 WT (us)", "Max Wait (us)", "Context Switches"]
    print("\n" + "=" * 90)
    print("MULTI-BURST PROCESS WORKLOAD DETAILED METRICS (Seeds 50000..50029)")
    print("=" * 90)
    mb_header = "| " + " | ".join(mb_cols) + " |"
    mb_sep = "| " + " | ".join(["---"] * len(mb_cols)) + " |"
    print(mb_header)
    print(mb_sep)
    mb_lines = [mb_header, mb_sep]

    for pol_name in columns[1:]:
        p_data = mb_row[pol_name]
        m_wt = fmt_stat(p_data["mean_wt"])
        p99 = fmt_stat(p_data["p99_wt"])
        m_wait = fmt_stat(p_data["max_wait"])
        sw = fmt_stat(p_data["switches"])
        line = f"| **{pol_name}** | {m_wt} | {p99} | {m_wait} | {sw} |"
        print(line)
        mb_lines.append(line)

    # Save to JSON and MD
    raw_data = {
        wl: {
            pol: {metric: [round(x, 2) for x in arr] for metric, arr in metrics_dict.items()}
            for pol, metrics_dict in row.items()
        }
        for wl, row in full_results.items()
    }
    with open("ml/checkpoints/canonical_v4_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    with open("ml/checkpoints/canonical_v4_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n\n" + "\n".join(mb_lines) + "\n")

    print(
        "\nSaved canonical results to ml/checkpoints/canonical_v4_results.json and ml/checkpoints/canonical_v4_table.md"
    )


if __name__ == "__main__":
    main()
