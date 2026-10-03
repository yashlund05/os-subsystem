"""Vectorized parallel environment wrapper for synchronous trajectory collection."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from userspace.trainer.env import SchedulerEnv


class VectorSchedulerEnv:
    """
    Synchronous vector environment wrapping N independent SchedulerEnv instances.
    Provides batched reset() and step().
    """

    def __init__(self, env_fns: List[Callable[..., SchedulerEnv]]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.top_k = self.envs[0].top_k

    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resets all environments.
        Returns:
            candidates: (num_envs, top_k, 16)
            action_masks: (num_envs, top_k)
        """
        cands_list = []
        masks_list = []
        for i, env in enumerate(self.envs):
            seed = seeds[i] if seeds is not None else None
            obs, _ = env.reset(seed=seed)
            cands_list.append(obs["candidates"])
            masks_list.append(obs["action_mask"])

        return np.stack(cands_list, axis=0), np.stack(masks_list, axis=0)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Steps all environments synchronously. Auto-resets on termination/truncation.
        Returns:
            next_candidates: (num_envs, top_k, 16)
            next_action_masks: (num_envs, top_k)
            rewards: (num_envs,)
            terminated: (num_envs,)
            truncated: (num_envs,)
            infos: List of info dicts
        """
        cands_list = []
        masks_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terms = np.zeros(self.num_envs, dtype=bool)
        truncs = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            act = int(actions[i])
            obs, r, term, trunc, info = env.step(act)
            rewards[i] = r
            terms[i] = term
            truncs[i] = trunc

            if term or trunc:
                # Save terminal observation and episode metrics in info
                info["terminal_observation"] = obs
                obs, _ = env.reset()

            cands_list.append(obs["candidates"])
            masks_list.append(obs["action_mask"])
            infos.append(info)

        return (
            np.stack(cands_list, axis=0),
            np.stack(masks_list, axis=0),
            rewards,
            terms,
            truncs,
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
