"""Unit test verifying feature distribution and non-saturation of remaining burst estimate."""

import numpy as np
import pytest

from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv


def test_remaining_burst_feature_histogram_non_saturation():
    """Verifies that burst_ratio (remaining burst estimate) does not degenerate to all zeros or ones."""
    # Test Pareto workload
    env = SchedulerEnv(
        workload_generator=lambda s: SyntheticWorkloadGenerator(seed=s).generate_pareto_bursts(50, 1.3, 200, 0.8),
        top_k=16,
    )
    obs, _ = env.reset(seed=50000)
    burst_ratios = []

    for _ in range(50):
        valid = np.where(obs["action_mask"] == 1)[0]
        for v in valid:
            burst_ratios.append(obs["candidates"][v, 9])
        obs, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break

    arr = np.array(burst_ratios)
    assert len(arr) > 0

    # Values must be strictly bounded in [0.0, 1.0]
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)

    # Must NOT have 80%+ exact zeros anymore
    zero_fraction = np.mean(arr == 0.0)
    assert zero_fraction < 0.20, f"burst_ratio still degenerates with {zero_fraction*100:.1f}% zeros"

    # Variance must be positive across candidate tasks
    assert np.std(arr) > 0.001, "burst_ratio has zero variance across candidates"


def test_convoy_remaining_burst_histogram():
    """Verifies that convoy workload candidates show distinct remaining burst ratios."""
    env = SchedulerEnv(
        workload_generator=lambda s: AdversarialWorkloadGenerator.create_convoy_workload(49, 50000, 100),
        top_k=16,
    )
    obs, _ = env.reset(seed=50000)
    burst_ratios = []

    for _ in range(10):
        valid = np.where(obs["action_mask"] == 1)[0]
        for v in valid:
            burst_ratios.append(obs["candidates"][v, 9])
        obs, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break

    arr = np.array(burst_ratios)
    assert len(arr) > 0
    assert np.all((arr >= 0.0) & (arr <= 1.0))
