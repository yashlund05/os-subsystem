"""Unit tests for the uniprocessor scheduling simulator and task representations."""

import pytest
from simulator.scheduling.task import SimulatedTask, TaskState
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from ml.models.policy_interface import DummyReferencePolicy, ModelType


def test_simulated_task_lifecycle():
    task = SimulatedTask(
        pid=101,
        arrival_time_us=1000,
        total_burst_us=5000
    )
    assert task.pid == 101
    assert task.remaining_burst_us == 5000
    assert task.state == TaskState.READY
    assert task.turnaround_time_us is None

    # Simulate execution
    task.first_dispatch_time_us = 1200
    task.executed_burst_us = 5000
    task.remaining_burst_us = 0
    task.completion_time_us = 6500
    task.state = TaskState.COMPLETED

    assert task.turnaround_time_us == 5500
    assert task.normalized_turnaround_time == 5500 / 5000
    assert task.response_time_us == 200


def test_synthetic_workload_generator():
    gen = SyntheticWorkloadGenerator(seed=12345)
    tasks = gen.generate_pareto_bursts(num_tasks=10, alpha=1.5, min_burst_us=200)

    assert len(tasks) == 10
    assert tasks[0].pid == 1
    assert tasks[-1].pid == 10
    # Arrivals should be monotonically non-decreasing
    arrivals = [t.arrival_time_us for t in tasks]
    assert arrivals == sorted(arrivals)
    # Burst durations must be at least min_burst_us
    for t in tasks:
        assert t.total_burst_us >= 200


def test_dummy_reference_policy():
    policy = DummyReferencePolicy()
    assert policy.model_type == ModelType.FP32_REFERENCE

    # Test evaluation on 2 tasks
    features = [
        [100.0] + [0.0] * 15,
        [500.0] + [0.0] * 15
    ]
    decisions = policy.evaluate(features)
    assert len(decisions) == 2
    # Shorter remaining burst should get higher priority score
    assert decisions[0].priority_score > decisions[1].priority_score
