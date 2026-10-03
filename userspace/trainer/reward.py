"""Configurable multi-objective reward calculator for CPU scheduling RL agents."""

from dataclasses import dataclass
from typing import List, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from simulator.scheduling.task import SimulatedTask


@dataclass
class RewardConfig:
    """Configurable weights and ablation toggles for step reward calculation."""

    w_wait: float = (
        1.0  # Normalized waiting time accumulation penalty (Little's Law: -(N_waiting*dt)/T_norm)
    )
    w_completion: float = 0.0  # Dropped completion bonus per user directive
    w_switch: float = 0.02  # Context-switch penalty rescaled to new Little's law magnitudes
    w_starvation: float = 0.1  # Max wait starvation penalty
    w_tail_threshold: float = 0.1  # Threshold proxy penalty for high waiting time
    w_invalid_action: float = 1.0  # Penalty for picking masked/empty slot

    # Ablation toggles
    enable_wait_penalty: bool = True
    enable_completion_bonus: bool = False  # Dropped per user directive
    enable_switch_penalty: bool = True
    enable_starvation_penalty: bool = True
    enable_tail_penalty: bool = True

    # Reference normalizers
    norm_step_us: float = (
        100000.0  # Fixed global constant T_norm (100ms) for Little's law: -(N_waiting*dt)/T_norm
    )
    norm_starve_wait_us: float = 50000.0


class RewardCalculator:
    """
    Computes fine-grained, independently toggleable rewards per scheduling step.
    Formula (Little's Law):
        R_t = - (w_wait * (N_waiting * dt) / T_norm + w_starve * starve_penalty + w_switch * switch_penalty)
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()

    def calculate_reward(
        self,
        step_elapsed_us: int,
        ready_tasks: List[SimulatedTask],
        num_completed: int,
        did_context_switch: bool,
        is_invalid_action: bool = False,
    ) -> float:
        reward = 0.0

        if is_invalid_action:
            reward -= self.config.w_invalid_action

        # 1. Little's-law waiting-time penalty: -(N_waiting * dt) / T_norm
        # Sums to -(Total_Waiting_Time / T_norm) over the entire episode (r = 1.0000 correlation with true wait time)
        if self.config.enable_wait_penalty and step_elapsed_us > 0:
            queue_len = len(ready_tasks)
            if queue_len > 0:
                wait_step_norm = (queue_len * step_elapsed_us) / self.config.norm_step_us
                reward -= self.config.w_wait * wait_step_norm

        # 2. Starvation penalty on max waiting time
        if self.config.enable_starvation_penalty and ready_tasks:
            max_wait = max((t.waiting_time_us for t in ready_tasks), default=0)
            if max_wait > self.config.norm_starve_wait_us:
                starve_ratio = (
                    max_wait - self.config.norm_starve_wait_us
                ) / self.config.norm_starve_wait_us
                reward -= self.config.w_starvation * float(np.clip(starve_ratio, 0.0, 5.0))

        # 3. Context switch overhead penalty
        if self.config.enable_switch_penalty and did_context_switch:
            reward -= self.config.w_switch

        # 4. Completion bonus (dropped per directive, maintained for backward toggle compatibility)
        if self.config.enable_completion_bonus and num_completed > 0:
            reward += self.config.w_completion * num_completed

        # 5. Tail threshold proxy penalty
        if self.config.enable_tail_penalty and ready_tasks:
            long_waiters = sum(
                1 for t in ready_tasks if t.waiting_time_us > self.config.norm_starve_wait_us * 1.5
            )
            if long_waiters > 0:
                reward -= self.config.w_tail_threshold * (long_waiters / len(ready_tasks))

        return float(reward)
