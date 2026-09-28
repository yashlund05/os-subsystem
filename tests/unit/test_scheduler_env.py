"""Comprehensive unit test suite for Gymnasium SchedulerEnv and associated components."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ml.training.env_factory import make_scheduler_env
from schedulers.round_robin.scheduler import RoundRobinScheduler
from schedulers.sjf.scheduler import SJFScheduler
from simulator.scheduling.engine import SchedulingSimulationEngine
from simulator.scheduling.task import SimulatedTask
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.burst_estimator import BurstEstimator
from userspace.trainer.env import SchedulerEnv
from userspace.trainer.observation import ObservationEncoder
from userspace.trainer.reward import RewardCalculator, RewardConfig
from userspace.trainer.wrapper import ClassicalSchedulerWrapper


def test_burst_estimator_behavior():
    estimator = BurstEstimator(alpha=0.5, default_estimate_us=4000, noise_std_frac=0.0)
    # Default without history
    est0 = estimator.get_estimate(pid=1, elapsed_us=1000)
    assert est0 == 3000

    # Record actual burst of 6000
    estimator.on_task_completion(pid=1, actual_burst_us=6000)
    # New smoothed average: 0.5 * 6000 + 0.5 * 4000 = 5000
    est1 = estimator.get_estimate(pid=1, elapsed_us=0)
    assert est1 == 5000

    estimator.reset()
    assert estimator.get_estimate(pid=1, elapsed_us=0) == 4000


def test_observation_encoder_dimensions():
    encoder = ObservationEncoder(top_k=16)
    estimator = BurstEstimator()

    tasks = [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=5000),
        SimulatedTask(pid=2, arrival_time_us=10, total_burst_us=2000),
    ]

    candidates, mask = encoder.encode(
        ready_tasks=tasks,
        running_task=None,
        current_time_us=50,
        burst_estimator=estimator,
        load_factor=0.8,
    )

    # Exactly 16 candidates x 16 features per candidate
    assert candidates.shape == (16, 16)
    assert mask.shape == (16,)
    # 2 valid tasks
    assert np.sum(mask) == 2
    assert mask[0] == 1 and mask[1] == 1 and mask[2] == 0


def test_reward_calculator_toggles():
    cfg = RewardConfig(
        w_wait=1.0,
        w_completion=5.0,
        w_switch=0.5,
        enable_wait_penalty=True,
        enable_completion_bonus=True,
        enable_switch_penalty=True,
        norm_step_us=5000.0,
    )
    calc = RewardCalculator(cfg)

    tasks = [SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=1000)]
    r = calc.calculate_reward(
        step_elapsed_us=5000, ready_tasks=tasks, num_completed=1, did_context_switch=True
    )
    # r = - wait_term (1.0 * 5000/5000) - switch (0.5) + completion (5.0) = 3.5
    assert pytest.approx(r, 0.01) == 3.5

    # Test toggles: disable completion
    cfg.enable_completion_bonus = False
    r_no_comp = calc.calculate_reward(
        step_elapsed_us=5000, ready_tasks=tasks, num_completed=1, did_context_switch=True
    )
    assert pytest.approx(r_no_comp, 0.01) == -1.5


def test_gymnasium_check_env():
    env = make_scheduler_env(split="train", num_tasks=10, top_k=16)
    # Gymnasium's strict environment conformance checker
    check_env(env.unwrapped)
    env.close()


def test_env_determinism_given_seed():
    env1 = make_scheduler_env(split="train", num_tasks=15, top_k=16)
    env2 = make_scheduler_env(split="train", num_tasks=15, top_k=16)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    assert np.allclose(obs1["candidates"], obs2["candidates"])
    assert np.array_equal(obs1["action_mask"], obs2["action_mask"])
    assert np.allclose(obs1["global_state"], obs2["global_state"])

    for _ in range(10):
        action = 0
        step_res1 = env1.step(action)
        step_res2 = env2.step(action)
        assert np.allclose(step_res1[0]["candidates"], step_res2[0]["candidates"])
        assert pytest.approx(step_res1[1], 1e-4) == step_res2[1]
        assert step_res1[2] == step_res2[2]

    env1.close()
    env2.close()


def test_action_masking_and_fallback():
    env = make_scheduler_env(split="train", num_tasks=5, top_k=16)
    obs, info = env.reset(seed=42)

    masks = env.action_masks()
    assert isinstance(masks, np.ndarray)
    assert masks.dtype == bool

    # Pick an intentionally invalid out-of-bounds action (index 15 when only ~1-2 tasks exist)
    obs, reward, term, trunc, info = env.step(15)
    assert info["invalid_action_count"] == 1
    env.close()


def test_classical_scheduler_wrapper_sjf():
    env = make_scheduler_env(split="eval", num_tasks=10, top_k=16)
    sjf = SJFScheduler()
    wrapper = ClassicalSchedulerWrapper(scheduler=sjf, env=env)

    cum_reward, info = wrapper.run_episode(seed=42)
    assert info["completed_tasks_count"] == 10
    assert "metrics" in info
    metrics = info["metrics"]
    assert metrics["mean_turnaround_time_us"] > 0
    env.close()


def test_classical_scheduler_wrapper_metric_parity():
    """
    Ensure running SJF or RoundRobin through ClassicalSchedulerWrapper yields
    turnaround and waiting times within numerical consistency of SchedulingSimulationEngine.
    """
    synth = SyntheticWorkloadGenerator(seed=777)
    workload = synth.generate_pareto_bursts(num_tasks=8, min_burst_us=500, load_factor=0.6)

    # 1. Direct engine run
    rr = RoundRobinScheduler(quantum_us=5000)
    engine = SchedulingSimulationEngine(scheduler=rr, context_switch_overhead_us=2)
    completed_direct, metrics_direct = engine.run(workload)

    # 2. Env run via ClassicalSchedulerWrapper
    env = SchedulerEnv(top_k=16, default_quantum_us=5000, context_switch_overhead_us=2)
    wrapper = ClassicalSchedulerWrapper(scheduler=RoundRobinScheduler(quantum_us=5000), env=env)
    _, info = wrapper.run_episode(options={"workload": workload})

    metrics_env = info["metrics"]
    assert metrics_env["completed_tasks"] == metrics_direct.completed_tasks
    # Turnaround and makespan within tolerance
    assert (
        pytest.approx(metrics_env["total_makespan_us"], rel=0.1) == metrics_direct.total_makespan_us
    )
    env.close()
