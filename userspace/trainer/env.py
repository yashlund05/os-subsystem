"""Gymnasium-compatible CPU scheduling environment for uniprocessor OS subsystems."""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator.scheduling.engine import SchedulingSimulationEngine
from simulator.scheduling.task import SimulatedTask, TaskState
from userspace.trainer.burst_estimator import BurstEstimator
from userspace.trainer.observation import DEFAULT_TOP_K, NEUROOS_INPUT_DIM, ObservationEncoder
from userspace.trainer.reward import RewardCalculator, RewardConfig


class SchedulerEnv(gym.Env):
    """
    Gymnasium uniprocessor scheduling environment built on the discrete-event simulator engine.

    Observation space:
      - 'candidates': (K, 16) float32 per-candidate feature matrix (10 task + 6 global features)
      - 'action_mask': (K,) int8 (1 = runnable task present, 0 = empty padded slot)
      - 'global_state': (6,) float32 system context

    Action space:
      - Discrete(K): Pick index of candidate task to execute for next quantum
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        workload_generator: Optional[Callable[[int], List[SimulatedTask]]] = None,
        top_k: int = DEFAULT_TOP_K,
        default_quantum_us: int = 5000,
        context_switch_overhead_us: int = 2,
        load_factor: float = 0.8,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 5000,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.default_quantum_us = default_quantum_us
        self.context_switch_overhead_us = context_switch_overhead_us
        self.load_factor = load_factor
        self.workload_generator = workload_generator
        self.max_steps = max_steps

        self.reward_calculator = RewardCalculator(reward_config or RewardConfig())
        self.observation_encoder = ObservationEncoder(top_k=top_k)
        self.burst_estimator = BurstEstimator()

        # Action space: pick candidate 0 .. top_k-1
        self.action_space = spaces.Discrete(self.top_k)

        # Observation space: Dict containing candidates matrix and action mask
        self.observation_space = spaces.Dict(
            {
                "candidates": spaces.Box(
                    low=0.0, high=10.0, shape=(self.top_k, NEUROOS_INPUT_DIM), dtype=np.float32
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.top_k,), dtype=np.int8),
                "global_state": spaces.Box(low=0.0, high=10.0, shape=(6,), dtype=np.float32),
            }
        )

        # Internal state
        self.current_time_us = 0
        self.total_steps = 0
        self.total_context_switches = 0
        self.total_cpu_busy_time_us = 0
        self.invalid_action_count = 0
        self.last_switch_time_us = 0
        self.last_running_pid: Optional[int] = None

        self.pending_arrivals: List[SimulatedTask] = []
        self.ready_queue: List[SimulatedTask] = []
        self.running_task: Optional[SimulatedTask] = None
        self.completed_tasks: List[SimulatedTask] = []
        self.current_slice_remaining_us = 0

    def action_masks(self) -> np.ndarray:
        """Returns boolean array of shape (top_k,) indicating valid actions."""
        # Running task (if any) takes candidate index 0, followed by ready queue
        num_valid = min(self.top_k, len(self.ready_queue) + (1 if self.running_task else 0))
        mask = np.zeros(self.top_k, dtype=bool)
        if num_valid > 0:
            mask[:num_valid] = True
        return mask

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.burst_estimator.seed(seed)

        # Generate fresh workload
        workload: List[SimulatedTask] = []
        if options and "workload" in options:
            workload = options["workload"]
        elif self.workload_generator is not None:
            workload = self.workload_generator(seed if seed is not None else 42)

        self.pending_arrivals = sorted(
            [copy.deepcopy(t) for t in workload], key=lambda t: (t.arrival_time_us, t.pid)
        )

        self.current_time_us = 0
        self.total_steps = 0
        self.total_context_switches = 0
        self.total_cpu_busy_time_us = 0
        self.invalid_action_count = 0
        self.last_switch_time_us = 0
        self.last_running_pid = None

        self.ready_queue.clear()
        self.completed_tasks.clear()
        self.running_task = None
        self.current_slice_remaining_us = 0
        self.burst_estimator.reset()

        # Advance to first arrival
        self._admit_arrivals()
        if not self.ready_queue and self.pending_arrivals:
            self.current_time_us = self.pending_arrivals[0].arrival_time_us
            self._admit_arrivals()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.total_steps += 1
        num_candidates = len(self.ready_queue) + (1 if self.running_task else 0)

        is_invalid = False
        if action < 0 or action >= self.top_k or action >= num_candidates:
            is_invalid = True
            self.invalid_action_count += 1
            # Fallback safety: pick candidate 0 (running task or oldest ready task)
            action = 0

        # Map action index to task candidate
        selected_task: Optional[SimulatedTask] = None
        did_context_switch = False

        if num_candidates > 0:
            if self.running_task is not None:
                if action == 0:
                    selected_task = self.running_task
                else:
                    # Preempt running task and pick ready_queue[action - 1]
                    self.running_task.state = TaskState.READY
                    self.ready_queue.append(self.running_task)
                    selected_task = self.ready_queue.pop(action - 1)
            else:
                selected_task = self.ready_queue.pop(action)

        # Context switch accounting
        if selected_task is not None:
            if self.last_running_pid is not None and self.last_running_pid != selected_task.pid:
                self.current_time_us += self.context_switch_overhead_us
                self.total_context_switches += 1
                selected_task.context_switches += 1
                did_context_switch = True
                self.last_switch_time_us = self.current_time_us

            self.running_task = selected_task
            self.running_task.state = TaskState.RUNNING
            self.last_running_pid = selected_task.pid
            self.current_slice_remaining_us = min(
                self.default_quantum_us, selected_task.remaining_burst_us
            )

            if selected_task.first_dispatch_time_us is None:
                selected_task.first_dispatch_time_us = self.current_time_us

        # Advance discrete-event time until next event:
        # (1) Task completion, (2) Quantum expiration, or (3) Next pending arrival
        time_to_completion = self.running_task.remaining_burst_us if self.running_task else 1
        time_to_quantum_end = (
            self.current_slice_remaining_us
            if self.current_slice_remaining_us > 0
            else time_to_completion
        )
        step_duration = min(time_to_completion, time_to_quantum_end)

        # Also stop at arrival so agent can make preemption decision if desired
        if self.pending_arrivals:
            time_to_arrival = max(
                0, self.pending_arrivals[0].arrival_time_us - self.current_time_us
            )
            if time_to_arrival > 0:
                step_duration = min(step_duration, time_to_arrival)

        step_duration = max(1, step_duration)

        # Advance simulation clock
        self.current_time_us += step_duration
        if self.running_task:
            self.total_cpu_busy_time_us += step_duration
            self.running_task.executed_burst_us += step_duration
            self.running_task.remaining_burst_us -= step_duration
            self.current_slice_remaining_us -= step_duration

        # Accumulate waiting time for all ready queue tasks
        for t in self.ready_queue:
            t.waiting_time_us += step_duration

        # Admit any newly arrived tasks
        self._admit_arrivals()

        # Check completion
        num_completed = 0
        if self.running_task and self.running_task.remaining_burst_us <= 0:
            self.running_task.state = TaskState.COMPLETED
            self.running_task.completion_time_us = self.current_time_us
            tt = self.running_task.turnaround_time_us
            assert tt is not None
            self.running_task.waiting_time_us = tt - self.running_task.total_burst_us
            self.burst_estimator.on_task_completion(
                self.running_task.pid, self.running_task.total_burst_us
            )
            self.completed_tasks.append(self.running_task)
            self.running_task = None
            self.current_slice_remaining_us = 0
            num_completed += 1

        # Check quantum expiration
        elif self.current_slice_remaining_us <= 0 and self.running_task:
            self.running_task.state = TaskState.READY
            # Priority demotion on quantum exhaustion (for multi-level feedback tracking)
            self.running_task.priority_level = min(3, self.running_task.priority_level + 1)
            self.ready_queue.append(self.running_task)
            self.running_task = None
            self.current_slice_remaining_us = 0

        # Fast forward if CPU is idle
        if self.running_task is None and not self.ready_queue and self.pending_arrivals:
            idle_gap = self.pending_arrivals[0].arrival_time_us - self.current_time_us
            if idle_gap > 0:
                self.current_time_us += idle_gap
            self._admit_arrivals()

        # Compute decomposed reward
        reward = self.reward_calculator.calculate_reward(
            step_elapsed_us=step_duration,
            ready_tasks=self.ready_queue,
            num_completed=num_completed,
            did_context_switch=did_context_switch,
            is_invalid_action=is_invalid,
        )

        terminated = (
            (not self.pending_arrivals) and (not self.ready_queue) and (self.running_task is None)
        )
        truncated = self.total_steps >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        if terminated or truncated:
            info["metrics"] = self._compute_episode_metrics()

        return obs, reward, terminated, truncated, info

    def _admit_arrivals(self) -> None:
        while (
            self.pending_arrivals
            and self.pending_arrivals[0].arrival_time_us <= self.current_time_us
        ):
            task = self.pending_arrivals.pop(0)
            task.state = TaskState.READY
            self.ready_queue.append(task)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        candidates, action_mask = self.observation_encoder.encode(
            ready_tasks=self.ready_queue,
            running_task=self.running_task,
            current_time_us=self.current_time_us,
            burst_estimator=self.burst_estimator,
            load_factor=self.load_factor,
            total_context_switches=self.total_context_switches,
            last_switch_time_us=self.last_switch_time_us,
        )
        # Global state vector
        global_state = np.array(
            [
                float(np.clip(len(self.ready_queue) / 1024.0, 0.0, 5.0)),
                float(np.clip(self.load_factor, 0.0, 2.0)),
                1.0 if self.running_task is not None else 0.0,
                float(
                    np.clip(
                        (self.running_task.executed_burst_us if self.running_task else 0)
                        / 100000.0,
                        0.0,
                        5.0,
                    )
                ),
                float(
                    np.clip(self.total_cpu_busy_time_us / max(1, self.current_time_us), 0.0, 1.0)
                ),
                float(np.clip(self.total_context_switches / 100.0, 0.0, 10.0)),
            ],
            dtype=np.float32,
        )

        return {"candidates": candidates, "action_mask": action_mask, "global_state": global_state}

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_time_us": self.current_time_us,
            "total_steps": self.total_steps,
            "ready_queue_len": len(self.ready_queue),
            "completed_tasks_count": len(self.completed_tasks),
            "total_context_switches": self.total_context_switches,
            "invalid_action_count": self.invalid_action_count,
        }

    def _compute_episode_metrics(self) -> Dict[str, float]:
        engine = SchedulingSimulationEngine(
            scheduler=None, context_switch_overhead_us=self.context_switch_overhead_us
        )
        summary = engine._compute_metrics(
            completed_tasks=self.completed_tasks,
            total_makespan_us=max(1, self.current_time_us),
            total_cpu_busy_time_us=self.total_cpu_busy_time_us,
            total_context_switches=self.total_context_switches,
        )
        return summary.to_dict()

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
