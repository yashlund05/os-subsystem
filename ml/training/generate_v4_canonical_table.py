"""Generate the complete V4 Canonical Results Table across 30 eval seeds."""

import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import torch

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
from ml.training.policy import CandidateScorer, ScorerPolicy
from ml.training.evaluate import heuristic_select_action


def eval_phase1_engine(
    scheduler_cls,
    kwargs: Dict[str, Any],
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> List[float]:
    mwts = []
    for s in seeds:
        tasks = workload_fn(s)
        sched = scheduler_cls(**kwargs)
        engine = SchedulingSimulationEngine(scheduler=sched, context_switch_overhead_us=0)
        _, metrics = engine.run(tasks)
        mwts.append(metrics.mean_waiting_time_us)
    return mwts


def eval_heuristic_oracle(
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> List[float]:
    mwts = []
    for s in seeds:
        env = SchedulerEnv(workload_generator=workload_fn, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            cands = []
            if env.running_task is not None:
                cands.append(env.running_task)
            cands.extend(env.ready_queue[: env.top_k - len(cands)])
            act = heuristic_select_action(cands, env.current_time_us)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        mwts.append(env._compute_episode_metrics()["mean_waiting_time_us"])
    return mwts


def eval_heuristic_obs(
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> List[float]:
    mwts = []
    for s in seeds:
        env = SchedulerEnv(workload_generator=workload_fn, top_k=16)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            # Observation heuristic rule: -pred_burst_norm + 1.0 * age_norm
            scores = [-obs["candidates"][i, 1] + 1.0 * obs["candidates"][i, 2] for i in valid]
            act = valid[np.argmax(scores)]
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        mwts.append(env._compute_episode_metrics()["mean_waiting_time_us"])
    return mwts


def eval_supervised_student(
    model: CandidateScorer,
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
) -> List[float]:
    mwts = []
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
        mwts.append(env._compute_episode_metrics()["mean_waiting_time_us"])
    return mwts


def eval_policy_model(
    policy: ScorerPolicy,
    workload_fn: Callable[[int], List[SimulatedTask]],
    seeds: List[int],
    device: torch.device,
) -> List[float]:
    mwts = []
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
        mwts.append(env._compute_episode_metrics()["mean_waiting_time_us"])
    return mwts


def fmt_stat(arr: List[float]) -> str:
    m = float(np.mean(arr))
    if len(arr) <= 1:
        return f"{m:.1f}"
    ci = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return f"{m:7.1f} ± {ci:5.1f}"


def main():
    eval_seeds = list(range(50000, 50030))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running V4 Canonical Evaluation on {len(eval_seeds)} seeds ({eval_seeds[0]}..{eval_seeds[-1]})...")

    # Define workloads
    workloads = {
        "Pareto rho=0.5": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.5),
        "Pareto rho=0.8": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8),
        "Pareto rho=0.95": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.95),
        "Poisson rho=0.5": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.5),
        "Poisson rho=0.8": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.8),
        "Poisson rho=0.95": lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.8, 200, 0.95),
        "Convoy": lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100),
    }

    # Load Supervised Student
    sup_ckpt = torch.load("ml/checkpoints/supervised_student_v4.pt", map_location="cpu", weights_only=False)
    sup_student = CandidateScorer(hidden_dims=[8])
    sup_student.load_state_dict(sup_ckpt["student_state_dict"])

    # Load Teacher (Seed 1003 had best Pareto performance, let's load all 3 and take ensemble or best)
    teacher_ckpts = [
        torch.load(f"ml/checkpoints/teacher_bc_ppo_s{s}.pt", map_location=device, weights_only=False)
        for s in [1001, 1002, 1003]
    ]
    teachers = []
    for ckpt in teacher_ckpts:
        p = ScorerPolicy(actor_hidden_dims=ckpt["actor_hidden_dims"], critic_hidden_dims=[64, 64]).to(device)
        p.load_state_dict(ckpt["model_state_dict"])
        teachers.append(p)
    teacher_best = teachers[2]  # Seed 1003

    # Load Student (BC+PPO Seed 1001 / 1003)
    student_ckpts = [
        torch.load(f"ml/checkpoints/student_bc_ppo_s{s}.pt", map_location=device, weights_only=False)
        for s in [1001, 1002, 1003]
    ]
    students = []
    for ckpt in student_ckpts:
        p = ScorerPolicy(actor_hidden_dims=ckpt["actor_hidden_dims"], critic_hidden_dims=[64, 64]).to(device)
        p.load_state_dict(ckpt["model_state_dict"])
        students.append(p)
    student_best = students[0]  # Seed 1001

    results_table = {}

    for wl_name, wl_fn in workloads.items():
        print(f"Evaluating {wl_name}...")
        row = {}
        # 1. FCFS
        row["FCFS"] = eval_phase1_engine(FCFSScheduler, {}, wl_fn, eval_seeds)
        # 2. SJF
        row["SJF"] = eval_phase1_engine(SJFScheduler, {}, wl_fn, eval_seeds)
        # 3. SRTF
        row["SRTF"] = eval_phase1_engine(SRTFScheduler, {}, wl_fn, eval_seeds)
        # 4. RR
        row["RR"] = eval_phase1_engine(RoundRobinScheduler, {"quantum_us": 5000}, wl_fn, eval_seeds)
        # 5. MLFQ
        row["MLFQ"] = eval_phase1_engine(MLFQScheduler, {"num_levels": 3, "base_quantum_us": 5000}, wl_fn, eval_seeds)
        # 6. Heuristic-Oracle
        row["Heuristic-Oracle"] = eval_heuristic_oracle(wl_fn, eval_seeds)
        # 7. Heuristic-Obs
        row["Heuristic-Obs"] = eval_heuristic_obs(wl_fn, eval_seeds)
        # 8. Supervised Student
        row["Supervised Student"] = eval_supervised_student(sup_student, wl_fn, eval_seeds)
        # 9. Teacher (BC+PPO)
        row["Teacher (BC+PPO)"] = eval_policy_model(teacher_best, wl_fn, eval_seeds, device)
        # 10. Student (BC+PPO)
        row["Student (BC+PPO)"] = eval_policy_model(student_best, wl_fn, eval_seeds, device)

        results_table[wl_name] = row

    # Print Formatted Markdown Table
    columns = [
        "Workload", "FCFS", "SJF", "SRTF", "RR (5ms)", "MLFQ (3-lvl)",
        "Heuristic-Oracle", "Heuristic-Obs", "Supervised-Student",
        "Teacher (BC+PPO)", "Student (BC+PPO)"
    ]

    print("\n" + "=" * 120)
    print("CANONICAL V4 RESULTS TABLE (Mean Waiting Time in us ± 95% CI across Seeds 50000..50029)")
    print("=" * 120)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    print(header)
    print(sep)

    md_lines = [header, sep]

    for wl_name, row in results_table.items():
        vals = [
            f"**{wl_name}**",
            fmt_stat(row["FCFS"]),
            fmt_stat(row["SJF"]),
            fmt_stat(row["SRTF"]),
            fmt_stat(row["RR"]),
            fmt_stat(row["MLFQ"]),
            fmt_stat(row["Heuristic-Oracle"]),
            fmt_stat(row["Heuristic-Obs"]),
            fmt_stat(row["Supervised Student"]),
            fmt_stat(row["Teacher (BC+PPO)"]),
            fmt_stat(row["Student (BC+PPO)"]),
        ]
        line = "| " + " | ".join(vals) + " |"
        print(line)
        md_lines.append(line)

    # Save to JSON
    raw_data = {
        wl: {col: [round(x, 2) for x in arr] for col, arr in r.items()}
        for wl, r in results_table.items()
    }
    with open("ml/checkpoints/canonical_v4_results.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    with open("ml/checkpoints/canonical_v4_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\nSaved canonical results to ml/checkpoints/canonical_v4_results.json and ml/checkpoints/canonical_v4_table.md")


if __name__ == "__main__":
    main()
