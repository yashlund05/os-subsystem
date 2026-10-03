"""Custom GPU-accelerated PPO algorithm with action masking, GAE, and decoupled actor-critic."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml.training.policy import ScorerPolicy
from ml.training.vec_env import VectorSchedulerEnv


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef_start: float = 0.02
    entropy_coef_end: float = 0.001
    max_grad_norm: float = 0.5
    num_rollout_steps: int = 128
    num_epochs: int = 4
    batch_size: int = 64
    target_kl: Optional[float] = 0.02


class PPOTrainer:
    """
    PPO trainer for discrete scheduling environment with action masking.
    """

    def __init__(
        self,
        policy: ScorerPolicy,
        vec_env: VectorSchedulerEnv,
        config: PPOConfig,
        device: torch.device,
    ) -> None:
        self.policy = policy.to(device)
        self.vec_env = vec_env
        self.cfg = config
        self.device = device

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
        self.total_timesteps = 0
        self.total_updates = 0

    def get_entropy_coef(self, progress: float) -> float:
        """Linear decay of entropy coefficient from start to end based on training progress in [0, 1]."""
        return self.cfg.entropy_coef_start + progress * (
            self.cfg.entropy_coef_end - self.cfg.entropy_coef_start
        )

    def collect_rollouts(
        self,
        obs_cands: np.ndarray,
        obs_masks: np.ndarray,
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray, List[float]]:
        """
        Collects num_rollout_steps across all vector environments.
        Returns rollout buffers, terminal obs, terminal masks, and completed episode returns.
        """
        num_envs = self.vec_env.num_envs
        steps = self.cfg.num_rollout_steps
        top_k = self.vec_env.top_k

        # Preallocate rollout buffers
        b_cands = torch.zeros((steps, num_envs, top_k, 16), dtype=torch.float32, device=self.device)
        b_masks = torch.zeros((steps, num_envs, top_k), dtype=torch.int8, device=self.device)
        b_actions = torch.zeros((steps, num_envs), dtype=torch.int64, device=self.device)
        b_logprobs = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
        b_rewards = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
        b_dones = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)
        b_values = torch.zeros((steps, num_envs), dtype=torch.float32, device=self.device)

        completed_returns: List[float] = []
        curr_ep_returns = np.zeros(num_envs, dtype=np.float32)

        curr_cands = obs_cands
        curr_masks = obs_masks

        with torch.no_grad():
            for t in range(steps):
                t_cands = torch.from_numpy(curr_cands).to(self.device)
                t_masks = torch.from_numpy(curr_masks).to(self.device)

                actions, logprobs, values = self.policy.act(t_cands, t_masks, deterministic=False)

                next_cands, next_masks, rewards, terms, truncs, infos = self.vec_env.step(
                    actions.cpu().numpy()
                )

                dones = np.logical_or(terms, truncs).astype(np.float32)
                curr_ep_returns += rewards
                for i in range(num_envs):
                    if dones[i]:
                        completed_returns.append(float(curr_ep_returns[i]))
                        curr_ep_returns[i] = 0.0

                b_cands[t] = t_cands
                b_masks[t] = t_masks
                b_actions[t] = actions
                b_logprobs[t] = logprobs
                b_rewards[t] = torch.from_numpy(rewards).to(self.device)
                b_dones[t] = torch.from_numpy(dones).to(self.device)
                b_values[t] = values

                curr_cands = next_cands
                curr_masks = next_masks

            # Bootstrap value for next state
            t_next_cands = torch.from_numpy(curr_cands).to(self.device)
            t_next_masks = torch.from_numpy(curr_masks).to(self.device)
            next_values = self.policy.get_value(t_next_cands, t_next_masks).squeeze(-1)

        # GAE calculation
        b_advantages = torch.zeros_like(b_rewards)
        last_gae: Any = 0.0
        for t in reversed(range(steps)):
            if t == steps - 1:
                next_non_terminal = 1.0 - torch.from_numpy(dones).to(self.device)
                next_v = next_values
            else:
                next_non_terminal = 1.0 - b_dones[t + 1]
                next_v = b_values[t + 1]

            delta = b_rewards[t] + self.cfg.gamma * next_v * next_non_terminal - b_values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
            b_advantages[t] = last_gae

        b_returns = b_advantages + b_values

        buffers = {
            "candidates": b_cands.view(-1, top_k, 16),
            "action_mask": b_masks.view(-1, top_k),
            "actions": b_actions.view(-1),
            "logprobs": b_logprobs.view(-1),
            "advantages": b_advantages.view(-1),
            "returns": b_returns.view(-1),
            "values": b_values.view(-1),
        }

        self.total_timesteps += steps * num_envs
        return buffers, curr_cands, curr_masks, completed_returns

    def update(
        self,
        buffers: Dict[str, torch.Tensor],
        progress: float = 0.0,
    ) -> Dict[str, float]:
        """
        Runs PPO update over minibatch epochs.
        """
        advantages = buffers["advantages"]
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        total_samples = buffers["actions"].size(0)
        batch_size = min(self.cfg.batch_size, total_samples)
        ent_coef = self.get_entropy_coef(progress)

        pg_losses, v_losses, ent_losses, approx_kls = [], [], [], []

        for _ in range(self.cfg.num_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                mb_cands = buffers["candidates"][batch_idx]
                mb_masks = buffers["action_mask"][batch_idx]
                mb_actions = buffers["actions"][batch_idx]
                mb_old_logprobs = buffers["logprobs"][batch_idx]
                mb_returns = buffers["returns"][batch_idx]
                mb_old_values = buffers["values"][batch_idx]
                mb_advantages = norm_advantages[batch_idx]

                new_values, new_logprobs, entropy = self.policy.evaluate_actions(
                    mb_cands, mb_masks, mb_actions
                )

                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())

                # Policy loss with PPO clip
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
                    * mb_advantages
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -self.cfg.clip_epsilon, self.cfg.clip_epsilon
                )
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy loss (entropy is positive; want to maximize entropy => minimize -entropy)
                ent_loss = -entropy.mean()

                loss = pg_loss + self.cfg.value_coef * v_loss + ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(-ent_loss.item())

            # Early stopping on KL divergence if exceeded target
            if self.cfg.target_kl is not None and np.mean(approx_kls) > self.cfg.target_kl * 1.5:
                break

        self.total_updates += 1
        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "entropy_coef": float(ent_coef),
        }
