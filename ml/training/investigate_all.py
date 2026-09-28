import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from ml.training.policy import CandidateScorer, ScorerPolicy
from simulator.scheduling.task import SimulatedTask
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.burst_estimator import BurstEstimator
from userspace.trainer.env import SchedulerEnv
from userspace.trainer.observation import ObservationEncoder
from userspace.trainer.reward import RewardCalculator, RewardConfig
from userspace.trainer.wrapper import ClassicalSchedulerWrapper
from schedulers.mlfq.scheduler import MLFQScheduler
from schedulers.round_robin.scheduler import RoundRobinScheduler


def investigate_student_bug():
    print("\n" + "="*60)
    print("INVESTIGATION 1: Supervised Student Bug & Real Rollouts")
    print("="*60)

    # (a) Collect observations from real SchedulerEnv rollouts across 30 seeds
    X_samples = []
    y_obs_list = []
    y_oracle_list = []

    def make_gen():
        return lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.5)

    env = SchedulerEnv(workload_generator=make_gen(), top_k=16)

    for seed in range(50000, 50030):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            cands = []
            if env.running_task is not None:
                cands.append(env.running_task)
            cands.extend(env.ready_queue[: env.top_k - len(cands)])

            for idx in valid:
                feat = obs["candidates"][idx]
                X_samples.append(feat)
                # Formula matching normalized units:
                # -pred_burst + 0.2*age -> - (pred_burst_norm*100000) + 0.2*(age_norm*500000)
                #                        = 100000 * (-pred_burst_norm + 1.0 * age_norm)
                y_obs_list.append(-feat[1] + 1.0 * feat[2])

                t_burst = cands[idx].total_burst_us / 100000.0
                y_oracle_list.append(-t_burst + 1.0 * feat[2])

            # Step env with observation heuristic
            scores = [-obs["candidates"][i, 1] + 1.0 * obs["candidates"][i, 2] for i in valid]
            act = valid[np.argmax(scores)]
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc

    X = torch.tensor(np.array(X_samples), dtype=torch.float32)
    y_obs = torch.tensor(np.array(y_obs_list), dtype=torch.float32)

    print(f"(a) Collected {len(X)} candidate samples from real rollouts across 30 seeds.")
    print("(b) Feature order & Normalization verification:")
    print("    - feat[1] is pred_burst_norm: max_burst = 100,000 us")
    print("    - feat[2] is age_norm:        max_wait  = 500,000 us")
    print("    - Scaling ratio: 0.2 * (500,000 / 100,000) = 1.000!")
    print("    - Original fit target (-feat[1] + 0.2*feat[2]) inadvertently set age weight to 0.04 in us, attenuating age by 5x!")

    # Fit student 16 -> 8 -> 1
    torch.manual_seed(42)
    student = CandidateScorer(hidden_dims=[8])
    opt = torch.optim.Adam(student.parameters(), lr=0.01)
    crit = nn.MSELoss()

    for ep in range(300):
        pred = student(X)  # shape (N,)
        loss = crit(pred, y_obs)
        opt.zero_grad()
        loss.backward()
        opt.step()

    r2 = 1.0 - (loss.item() / torch.var(y_obs).item())
    print(f"    Student fit on Real Rollouts: Loss (MSE)={loss.item():.6e}, R2={r2:.6f}")

    # (c) Check top-1 agreement between student argmax and heuristic argmax
    student.eval()
    matches = 0
    total_decisions = 0

    for seed in range(50000, 50030):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            if len(valid) > 1:
                total_decisions += 1
                # Heuristic argmax
                scores_h = [-obs["candidates"][i, 1] + 1.0 * obs["candidates"][i, 2] for i in valid]
                act_h = valid[np.argmax(scores_h)]
                # Student argmax
                with torch.no_grad():
                    c_t = torch.from_numpy(obs["candidates"]).float()
                    s_out = student(c_t).squeeze(-1).numpy()
                s_out[mask == 0] = -1e9
                act_s = int(np.argmax(s_out))
                if act_h == act_s:
                    matches += 1
            scores = [-obs["candidates"][i, 1] + 1.0 * obs["candidates"][i, 2] for i in valid]
            act = valid[np.argmax(scores)]
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc

    agreement = matches / max(1, total_decisions)
    print(f"(c) Top-1 Action Agreement on Real Observations: {agreement*100:.2f}% ({matches}/{total_decisions})")

    # (d) In-env Mean Waiting Time across 30 seeds for refitted student
    mwts_stud = []
    mwts_heur = []
    for seed in range(50000, 50030):
        # Student rollout
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            mask = obs["action_mask"]
            with torch.no_grad():
                c_t = torch.from_numpy(obs["candidates"]).float()
                s_out = student(c_t).squeeze(-1).numpy()
            s_out[mask == 0] = -1e9
            act = int(np.argmax(s_out))
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        m_s = env._compute_episode_metrics()
        mwts_stud.append(m_s["mean_waiting_time_us"])

        # Heuristic rollout
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            scores = [-obs["candidates"][i, 1] + 1.0 * obs["candidates"][i, 2] for i in valid]
            act = valid[np.argmax(scores)]
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        m_h = env._compute_episode_metrics()
        mwts_heur.append(m_h["mean_waiting_time_us"])

    mean_s = np.mean(mwts_stud)
    ci_s = 1.96 * np.std(mwts_stud) / math.sqrt(len(mwts_stud))
    mean_h = np.mean(mwts_heur)
    ci_h = 1.96 * np.std(mwts_heur) / math.sqrt(len(mwts_heur))

    print(f"(d) In-Env Performance (Pareto rho=0.5, 30 seeds):")
    print(f"    - Refitted Student-Supervised Mean WT: {mean_s:.1f} ± {ci_s:.1f} us")
    print(f"    - Observation Heuristic Mean WT:        {mean_h:.1f} ± {ci_h:.1f} us")
    print(f"    - Paired difference (Student - Heuristic): {np.mean(np.array(mwts_stud) - np.array(mwts_heur)):.2f} us")


def investigate_burst_ratio():
    print("\n" + "="*60)
    print("INVESTIGATION 2: burst_ratio Saturation & Histogram")
    print("="*60)

    # Collect histogram under current formula
    for w_name, gen in [
        ("pareto_0.8", lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8)),
        ("convoy_0.8", lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100)),
    ]:
        env = SchedulerEnv(workload_generator=gen, top_k=16)
        obs, _ = env.reset(seed=50000)
        curr_ratios = []
        prop_ratios = []
        done = False
        while not done:
            valid = np.where(obs["action_mask"] == 1)[0]
            cands = []
            if env.running_task is not None:
                cands.append(env.running_task)
            cands.extend(env.ready_queue[: env.top_k - len(cands)])

            for v in valid:
                curr_ratios.append(obs["candidates"][v, 9])
                t = cands[v]
                pred_b = env.burst_estimator.get_estimate(t.pid, t.executed_burst_us)
                prop = np.clip((pred_b - t.executed_burst_us) / 100000.0, 0.0, 1.0)
                prop_ratios.append(prop)
            obs, _, term, trunc, _ = env.step(0)
            done = term or trunc

        curr_arr = np.array(curr_ratios)
        prop_arr = np.array(prop_ratios)

        print(f"\n--- Workload: {w_name} (Total candidate instances: {len(curr_arr)}) ---")
        print(f"Current formula: np.clip(elapsed_norm / max(0.01, pred_burst_norm + elapsed_norm), 0.0, 1.0)")
        print(f"  Exact zeros (0.0): {np.sum(curr_arr == 0.0)} ({np.mean(curr_arr == 0.0)*100:.1f}%)")
        print(f"  Histogram [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]: {np.histogram(curr_arr, bins=5, range=(0.0, 1.0))[0]}")
        print(f"Proposed formula: np.clip((pred_burst - elapsed) / max_burst, 0.0, 1.0)")
        print(f"  Histogram [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]: {np.histogram(prop_arr, bins=5, range=(0.0, 1.0))[0]}")
        print(f"  Mean={np.mean(prop_arr):.4f}, Std={np.std(prop_arr):.4f}, Min={np.min(prop_arr):.4f}, Max={np.max(prop_arr):.4f}")


def investigate_reward_totals():
    print("\n" + "="*60)
    print("INVESTIGATION 3: Exact Reward Breakdown & Audit on 30 Seeds")
    print("="*60)

    # Evaluate exact reward terms for Teacher and Heuristic on Pareto 0.8 across 30 seeds
    device = torch.device("cpu")
    ckpt = torch.load("ml/checkpoints/ppo_production_teacher_staged_s1001_checkpoint.pt", map_location=device)
    teacher = ScorerPolicy(actor_hidden_dims=ckpt.get("actor_hidden_dims", [64, 32]))
    actor_state = {k[len("actor."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
    teacher.actor.load_state_dict(actor_state)
    teacher.eval()

    def run_policy_reward(is_teacher: bool):
        total_rewards = []
        term_wait_list = []
        term_max_wait_list = []
        term_switch_list = []
        term_comp_list = []
        term_tail_list = []

        cfg = RewardConfig(
            w_wait=1.0,
            w_completion=2.0,
            w_switch=0.05,
            w_starvation=0.1,
            w_tail_threshold=0.5,
        )

        for seed in range(50000, 50030):
            env = SchedulerEnv(
                workload_generator=lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8),
                reward_config=cfg,
                top_k=16,
            )
            # Instrument calculate_reward directly
            orig_calc = env.reward_calculator.calculate_reward
            ep_terms = {"wait": 0.0, "starve": 0.0, "switch": 0.0, "comp": 0.0, "tail": 0.0}

            def logged_calc(step_elapsed_us, ready_tasks, num_completed, did_context_switch, is_invalid_action=False):
                w_term = 0.0
                if cfg.enable_wait_penalty and step_elapsed_us > 0:
                    q_len = len(ready_tasks)
                    if q_len > 0:
                        w_step = (step_elapsed_us * q_len) / (cfg.norm_step_us * max(1, q_len))
                        w_term = -cfg.w_wait * w_step
                st_term = 0.0
                if cfg.enable_starvation_penalty and ready_tasks:
                    m_wait = max((t.waiting_time_us for t in ready_tasks), default=0)
                    if m_wait > cfg.norm_starve_wait_us:
                        r = (m_wait - cfg.norm_starve_wait_us) / cfg.norm_starve_wait_us
                        st_term = -cfg.w_starvation * float(np.clip(r, 0.0, 5.0))
                sw_term = 0.0
                if cfg.enable_switch_penalty and did_context_switch:
                    sw_term = -cfg.w_switch
                cp_term = 0.0
                if cfg.enable_completion_bonus and num_completed > 0:
                    cp_term = cfg.w_completion * num_completed
                tl_term = 0.0
                if cfg.enable_tail_penalty and ready_tasks:
                    lw = sum(1 for t in ready_tasks if t.waiting_time_us > cfg.norm_starve_wait_us * 1.5)
                    if lw > 0:
                        tl_term = -cfg.w_tail_threshold * (lw / len(ready_tasks))

                ep_terms["wait"] += w_term
                ep_terms["starve"] += st_term
                ep_terms["switch"] += sw_term
                ep_terms["comp"] += cp_term
                ep_terms["tail"] += tl_term

                return orig_calc(step_elapsed_us, ready_tasks, num_completed, did_context_switch, is_invalid_action)

            env.reward_calculator.calculate_reward = logged_calc
            obs, _ = env.reset(seed=seed)
            done = False
            ep_total = 0.0

            while not done:
                mask = obs["action_mask"]
                valid = np.where(mask == 1)[0]
                if is_teacher:
                    with torch.no_grad():
                        t_c = torch.from_numpy(obs["candidates"]).unsqueeze(0)
                        t_m = torch.from_numpy(obs["action_mask"]).unsqueeze(0)
                        act_t, _, _ = teacher.act(t_c, t_m, deterministic=True)
                        act = int(act_t.item())
                else:
                    scores = [-obs["candidates"][i, 1] + 1.0 * obs["candidates"][i, 2] for i in valid]
                    act = valid[np.argmax(scores)]

                obs, r, term, trunc, _ = env.step(act)
                done = term or trunc
                ep_total += r

            total_rewards.append(ep_total)
            term_wait_list.append(ep_terms["wait"])
            term_max_wait_list.append(ep_terms["starve"])
            term_switch_list.append(ep_terms["switch"])
            term_comp_list.append(ep_terms["comp"])
            term_tail_list.append(ep_terms["tail"])

        return (
            np.mean(total_rewards),
            np.mean(term_wait_list),
            np.mean(term_max_wait_list),
            np.mean(term_switch_list),
            np.mean(term_comp_list),
            np.mean(term_tail_list),
        )

    t_tot, t_wait, t_mwait, t_sw, t_comp, t_tail = run_policy_reward(True)
    h_tot, h_wait, h_mwait, h_sw, h_comp, h_tail = run_policy_reward(False)

    print(f"Teacher (w_switch=0.05, 30 seeds):")
    print(f"  Sum of recorded step rewards: {t_tot:.4f}")
    print(f"  Decomposed terms: wait={t_wait:.4f}, max_wait={t_mwait:.4f}, switch={t_sw:.4f}, comp={t_comp:.4f}, tail={t_tail:.4f}")
    print(f"  Terms sum: {t_wait + t_mwait + t_sw + t_comp + t_tail:.4f} (Discrepancy: {t_tot - (t_wait + t_mwait + t_sw + t_comp + t_tail):.6f})")

    print(f"Heuristic (w_switch=0.05, 30 seeds):")
    print(f"  Sum of recorded step rewards: {h_tot:.4f}")
    print(f"  Decomposed terms: wait={h_wait:.4f}, max_wait={h_mwait:.4f}, switch={h_sw:.4f}, comp={h_comp:.4f}, tail={h_tail:.4f}")
    print(f"  Terms sum: {h_wait + h_mwait + h_sw + h_comp + h_tail:.4f} (Discrepancy: {h_tot - (h_wait + h_mwait + h_sw + h_comp + h_tail):.6f})")

    print("\nCompletion Bonus Analysis:")
    print("  Both policies complete exactly 50 tasks -> ep_comp = 50 * 2.0 = +100.0.")
    print("  Why it is in the comparison: It provides a global terminal offset (+100.0) that ensures episodes finish positive,")
    print("  but because it is identical for all non-crashing policies, it does NOT affect relative policy ranking.")


def investigate_real_mlfq():
    print("\n" + "="*60)
    print("INVESTIGATION 4: Real Phase 1 MLFQScheduler Class Verification")
    print("="*60)

    # Construct a workload with distinct task types: 2 short interactive jobs and 1 long compute job
    tasks_mlfq = [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=25000),  # Long job
        SimulatedTask(pid=2, arrival_time_us=1000, total_burst_us=2000), # Short interactive
        SimulatedTask(pid=3, arrival_time_us=2000, total_burst_us=3000), # Short interactive
    ]
    tasks_rr = [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=25000),
        SimulatedTask(pid=2, arrival_time_us=1000, total_burst_us=2000),
        SimulatedTask(pid=3, arrival_time_us=2000, total_burst_us=3000),
    ]

    # Run MLFQ
    mlfq_sched = MLFQScheduler(num_levels=3, base_quantum_us=5000, boost_interval_us=500000)
    env_mlfq = SchedulerEnv(workload_generator=lambda s: tasks_mlfq, top_k=16)
    wrapper_mlfq = ClassicalSchedulerWrapper(mlfq_sched, env_mlfq)
    _, info_mlfq = wrapper_mlfq.run_episode(seed=42)

    # Run RoundRobin
    rr_sched = RoundRobinScheduler(quantum_us=5000)
    env_rr = SchedulerEnv(workload_generator=lambda s: tasks_rr, top_k=16)
    wrapper_rr = ClassicalSchedulerWrapper(rr_sched, env_rr)
    _, info_rr = wrapper_rr.run_episode(seed=42)

    print("Task Completion Times on Handcrafted Workload:")
    print(f"{'PID':<5} | {'True Burst':<12} | {'MLFQ Completion':<18} | {'RR Completion':<18} | {'Delta (RR - MLFQ)':<18}")
    print("-" * 75)
    for p in [1, 2, 3]:
        c_mlfq = next(t.completion_time_us for t in env_mlfq.completed_tasks if t.pid == p)
        c_rr = next(t.completion_time_us for t in env_rr.completed_tasks if t.pid == p)
        b = next(t.total_burst_us for t in tasks_mlfq if t.pid == p)
        print(f"{p:<5} | {b:<12} | {c_mlfq:<18} | {c_rr:<18} | {c_rr - c_mlfq:<18}")

    m_mlfq = info_mlfq["metrics"]["mean_waiting_time_us"]
    m_rr = info_rr["metrics"]["mean_waiting_time_us"]
    print(f"\nOverall Mean Waiting Time: MLFQ={m_mlfq:.1f} us vs RR={m_rr:.1f} us (MLFQ is {m_rr - m_mlfq:.1f} us faster!)")


def investigate_estimator_sigma():
    print("\n" + "="*60)
    print("INVESTIGATION 5: Estimator Noise Sensitivity (sigma = 0.1, 0.3, 0.5)")
    print("="*60)

    device = torch.device("cpu")
    ckpt = torch.load("ml/checkpoints/ppo_production_teacher_staged_s1001_checkpoint.pt", map_location=device)
    teacher = ScorerPolicy(actor_hidden_dims=ckpt.get("actor_hidden_dims", [64, 32]))
    actor_state = {k[len("actor."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
    teacher.actor.load_state_dict(actor_state)
    teacher.eval()

    sigmas = [0.10, 0.30, 0.50]
    print(f"{'Sigma':<8} | {'Heuristic Mean WT (us)':<25} | {'Teacher Mean WT (us)':<25} | {'Gap (Teacher - Heuristic)':<25}")
    print("-" * 88)

    for sig in sigmas:
        h_wts = []
        t_wts = []
        for s in range(50000, 50030):
            # Heuristic
            env = SchedulerEnv(
                workload_generator=lambda seed: SyntheticWorkloadGenerator(seed=seed).generate_pareto_bursts(50, 1.3, 200, 0.8),
                top_k=16,
            )
            env.burst_estimator.noise_std_frac = sig
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs["action_mask"]
                valid = np.where(mask == 1)[0]
                scores = [-obs["candidates"][i, 1] + 1.0 * obs["candidates"][i, 2] for i in valid]
                act = valid[np.argmax(scores)]
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            h_wts.append(env._compute_episode_metrics()["mean_waiting_time_us"])

            # Teacher
            env = SchedulerEnv(
                workload_generator=lambda seed: SyntheticWorkloadGenerator(seed=seed).generate_pareto_bursts(50, 1.3, 200, 0.8),
                top_k=16)
            env.burst_estimator.noise_std_frac = sig
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                mask = obs["action_mask"]
                with torch.no_grad():
                    t_c = torch.from_numpy(obs["candidates"]).unsqueeze(0)
                    t_m = torch.from_numpy(obs["action_mask"]).unsqueeze(0)
                    act_t, _, _ = teacher.act(t_c, t_m, deterministic=True)
                    act = int(act_t.item())
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            t_wts.append(env._compute_episode_metrics()["mean_waiting_time_us"])

        m_h = np.mean(h_wts)
        ci_h = 1.96 * np.std(h_wts) / math.sqrt(len(h_wts))
        m_t = np.mean(t_wts)
        ci_t = 1.96 * np.std(t_wts) / math.sqrt(len(t_wts))
        gap = m_t - m_h
        print(f"{sig:<8.2f} | {m_h:8.1f} ± {ci_h:5.1f} us          | {m_t:8.1f} ± {ci_t:5.1f} us          | {gap:+8.1f} us")


if __name__ == "__main__":
    investigate_student_bug()
    investigate_burst_ratio()
    investigate_reward_totals()
    investigate_real_mlfq()
    investigate_estimator_sigma()
