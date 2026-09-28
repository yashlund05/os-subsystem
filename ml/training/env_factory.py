"""Environment factory module providing train/eval splits and workload distribution sampling."""

from typing import List, Optional

from simulator.scheduling.task import SimulatedTask
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from userspace.trainer.env import SchedulerEnv
from userspace.trainer.reward import RewardConfig

TRAIN_SEED_BASE = 1000
EVAL_SEED_BASE = 50000


def make_scheduler_env(
    split: str = "train",
    workload_type: str = "pareto",
    num_tasks: int = 50,
    load_factor: float = 0.8,
    top_k: int = 16,
    reward_config: Optional[RewardConfig] = None,
    seed: Optional[int] = None,
) -> SchedulerEnv:
    """
    Factory creating a SchedulerEnv with disjoint train/eval seed ranges.

    Args:
        split: 'train' (seed base 1000..49999) or 'eval' (seed base 50000..99999)
        workload_type: 'pareto', 'poisson', or 'convoy'
        num_tasks: Number of tasks generated per episode
        load_factor: Offered load rho in [0.10, 0.98]
        top_k: Maximum candidates exposed in action/observation space
    """
    if split == "train":
        seed_offset = TRAIN_SEED_BASE
    elif split == "eval":
        seed_offset = EVAL_SEED_BASE
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train' or 'eval'.")

    def generator(episode_seed: int) -> List[SimulatedTask]:
        effective_seed = (seed_offset + episode_seed) % 1000000
        if workload_type == "convoy":
            return AdversarialWorkloadGenerator.create_convoy_workload(
                num_short_jobs=num_tasks - 1, long_burst_us=50000, short_burst_us=100
            )
        else:
            synth = SyntheticWorkloadGenerator(seed=effective_seed)
            return synth.generate_pareto_bursts(
                num_tasks=num_tasks,
                alpha=1.3 if workload_type == "pareto" else 1.8,
                min_burst_us=200,
                load_factor=load_factor,
            )

    env = SchedulerEnv(
        workload_generator=generator,
        top_k=top_k,
        load_factor=load_factor,
        reward_config=reward_config,
    )
    return env
