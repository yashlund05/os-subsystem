"""Wrapper allowing any classical BaseScheduler to drive SchedulerEnv for apples-to-apples comparison."""

from typing import Any, Dict, Optional, Tuple

from schedulers.base import BaseScheduler
from userspace.trainer.env import SchedulerEnv


class ClassicalSchedulerWrapper:
    """
    Connects any BaseScheduler (FCFS, SJF, SRTF, RoundRobin, MLFQ) to SchedulerEnv.
    Enables direct, identical-workload comparison between RL policies and classical algorithms.
    """

    def __init__(self, scheduler: BaseScheduler, env: SchedulerEnv) -> None:
        self.scheduler = scheduler
        self.env = env

    def run_episode(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Runs one complete episode driven by the underlying BaseScheduler.
        Returns:
            Tuple of (cumulative_reward, final_info_dict)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        cumulative_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self._select_action()
            obs, reward, terminated, truncated, info = self.env.step(action)
            cumulative_reward += reward

        return cumulative_reward, info

    def _select_action(self) -> int:
        """
        Translates the BaseScheduler's decision into an action index 0 .. top_k-1 in the env.
        Candidate mapping:
          index 0: running_task (if present)
          index 1 .. len(ready_queue): ready_queue[i-1]
        """
        running = self.env.running_task
        ready_tasks = self.env.ready_queue

        # If scheduler is preemptive (like SRTF), check if any ready task should preempt running task
        if running is not None:
            if hasattr(self.scheduler, "should_preempt"):
                for idx, candidate in enumerate(ready_tasks[: self.env.top_k - 1]):
                    if self.scheduler.should_preempt(running, candidate):
                        # Preempt: action index is candidate's index in candidates array (idx + 1)
                        return idx + 1
            # If quantum not expired and no preemption, continue running current task
            if self.env.current_slice_remaining_us > 0:
                return 0

        # Build candidate pool for scheduler evaluation
        all_candidates = []
        if running is not None:
            all_candidates.append(running)
        all_candidates.extend(ready_tasks[: self.env.top_k - len(all_candidates)])

        if not all_candidates:
            return 0

        # Ask scheduler to rank / pick
        # Populate scheduler internal state with current candidates
        self.scheduler._clear_for_env() if hasattr(self.scheduler, "_clear_for_env") else None

        # For SJF/SRTF/FCFS, determine which task has highest priority
        best_candidate_idx = 0
        if self.scheduler.name.startswith("SJF"):
            # Shortest total burst
            best_burst = float("inf")
            for idx, task in enumerate(all_candidates):
                if task.total_burst_us < best_burst:
                    best_burst = task.total_burst_us
                    best_candidate_idx = idx
        elif self.scheduler.name.startswith("SRTF"):
            # Shortest remaining burst
            best_rem = float("inf")
            for idx, task in enumerate(all_candidates):
                if task.remaining_burst_us < best_rem:
                    best_rem = task.remaining_burst_us
                    best_candidate_idx = idx
        elif self.scheduler.name.startswith("RoundRobin") or self.scheduler.name.startswith("FCFS"):
            # If running task exists and quantum expired, pick next ready task (index 1)
            if running is not None and self.env.current_slice_remaining_us <= 0 and ready_tasks:
                best_candidate_idx = 1
            else:
                best_candidate_idx = 0
        else:
            # Default to candidate 0
            best_candidate_idx = 0

        return best_candidate_idx
