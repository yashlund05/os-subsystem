"""Configurable multi-objective reward calculator for CPU scheduling RL agents."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from simulator.scheduling.task import SimulatedTask


@dataclass
class RewardConfig:
    """Configurable weights and ablation toggles for step reward calculation."""

    w_wait: float = 1.0  # Normalized waiting time accumulation penalty
    w_completion: float = 2.0  # Bonus per completed task
    w_switch: float = 0.2  # Context-switch penalty
    w_starvation: float = 0.5  # Max wait starvation penalty
    w_tail_threshold: float = 0.3  # Threshold proxy penalty for high waiting time
    w_invalid_action: float = 1.0  # Penalty for picking masked/empty slot

    # Ablation toggles
    enable_wait_penalty: bool = True
    enable_completion_bonus: bool = True
    enable_switch_penalty: bool = True
    enable_starvation_penalty: bool = True
    enable_tail_penalty: bool = True

    # Reference normalizers
    norm_step_us: float = 5000.0
    norm_starve_wait_us: float = 50000.0


class RewardCalculator:
    """
    Computes fine-grained, independently toggleable rewards per scheduling step.
    Formula:
        R_t = - (w_wait * wait_penalty + w_starve * starve_penalty + w_switch * switch_penalty) + w_done * completions
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

        # 1. Waiting-time penalty: cumulative waiting time of runnable tasks during step_elapsed_us
        if self.config.enable_wait_penalty and step_elapsed_us > 0:
            queue_len = len(ready_tasks)
            if queue_len > 0:
                # Normalized by step duration and queue size
                wait_step_norm = (step_elapsed_us * queue_len) / (
                    self.config.norm_step_us * max(1, queue_len)
                )
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

        # 4. Completion bonus
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
