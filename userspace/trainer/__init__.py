"""Userspace Trainer Package for RL-based CPU Scheduling."""

from userspace.trainer.burst_estimator import BurstEstimator
from userspace.trainer.observation import ObservationEncoder
from userspace.trainer.reward import RewardCalculator, RewardConfig

try:
    from userspace.trainer.env import SchedulerEnv
    from userspace.trainer.wrapper import ClassicalSchedulerWrapper

    __all__ = [
        "SchedulerEnv",
        "ObservationEncoder",
        "RewardCalculator",
        "RewardConfig",
        "BurstEstimator",
        "ClassicalSchedulerWrapper",
    ]
except ImportError:  # Minimal env without gymnasium/torch (Phase 4 CPU-only CI)
    __all__ = [
        "ObservationEncoder",
        "RewardCalculator",
        "RewardConfig",
        "BurstEstimator",
    ]
