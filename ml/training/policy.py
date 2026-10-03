"""Neural network architectures for NeuroOS scheduling policies.

Provides:
- CandidateScorerNetwork: 16 -> hidden -> 1 shared MLP actor with masked softmax.
- CriticNetwork: Separate value network operating on pooled raw candidate features + global context.
- ScorerPolicy: Unified wrapper managing actor and critic forward passes, masked action selection,
  and log-probability evaluation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

FEATURE_NAMES = [
    # 10 task features
    "elapsed_norm",
    "pred_burst_norm",
    "age_norm",
    "ctx_switches_norm",
    "cache_miss_norm",
    "branch_mispred_norm",
    "mem_kb_norm",
    "priority_norm",
    "is_running_val",
    "burst_ratio",
    # 6 global features (inside candidate vector)
    "queue_len_norm",
    "load_factor_norm",
    "cpu_busy_frac",
    "max_wait_norm",
    "mean_pred_norm",
    "time_since_switch_norm",
]


class CandidateScorer(nn.Module):
    """
    Shared per-candidate scorer MLP (16 -> hidden -> 1).

    Applies identical weights to every candidate vector in X: (batch, top_k, 16) -> (batch, top_k).
    Only the actor weights from this module are exported to kernel micro-core.
    """

    def __init__(self, input_dim: int = 16, hidden_dims: Optional[List[int]] = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]  # Default teacher: 16 -> 64 -> 32 -> 1

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)

        layers: List[nn.Module] = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, top_k, 16) or (top_k, 16) or (16,)
        Returns:
            scores: Tensor of shape (batch, top_k) or (top_k,)
        """
        orig_shape = x.shape
        if len(orig_shape) == 1:
            return self.net(x.unsqueeze(0)).squeeze()
        elif len(orig_shape) == 2:
            # (top_k, 16)
            out = self.net(x)  # (top_k, 1)
            return out.squeeze(-1)
        elif len(orig_shape) == 3:
            # (batch, top_k, 16)
            b, k, d = orig_shape
            flat = x.view(b * k, d)
            scores = self.net(flat).view(b, k)
            return scores
        else:
            raise ValueError(f"Unsupported input shape: {orig_shape}")


class CriticNetwork(nn.Module):
    """
    Separate Critic network decoupled from Actor.
    Takes pooled candidate features (mean/max over valid candidates) + global context features
    and produces a scalar state-value estimate V(s).
    """

    def __init__(
        self,
        candidate_dim: int = 10,
        global_dim: int = 6,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.candidate_dim = candidate_dim
        self.global_dim = global_dim

        # Input to critic: pooled candidate features (mean 10 + max 10 = 20) + global features (6) = 26
        critic_in_dim = (candidate_dim * 2) + global_dim

        layers: List[nn.Module] = []
        curr_dim = critic_in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        candidates: torch.Tensor,
        action_mask: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            candidates: (batch, top_k, 16)
            action_mask: (batch, top_k) int8 or bool
            global_state: (batch, 6) optional if already extracted
        Returns:
            values: (batch, 1)
        """
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            if global_state is not None:
                global_state = global_state.unsqueeze(0)

        batch_size, top_k, _ = candidates.shape
        mask_f = action_mask.float().unsqueeze(-1)  # (batch, top_k, 1)

        task_features = candidates[:, :, : self.candidate_dim]  # (batch, top_k, 10)
        masked_tasks = task_features * mask_f

        # Mean pooling over valid candidates
        valid_counts = mask_f.sum(dim=1).clamp(min=1.0)  # (batch, 1)
        mean_pooled = masked_tasks.sum(dim=1) / valid_counts  # (batch, 10)

        # Max pooling over valid candidates
        neg_inf_mask = (1.0 - mask_f) * -1e9
        max_pooled = (masked_tasks + neg_inf_mask).max(dim=1).values
        # If no candidates valid, clamp to 0
        max_pooled = torch.where(valid_counts > 0, max_pooled, torch.zeros_like(max_pooled))

        # Extract global features (either from candidates[:, 0, 10:] or passed global_state)
        if global_state is None:
            global_state = candidates[:, 0, self.candidate_dim :]  # (batch, 6)

        critic_in = torch.cat([mean_pooled, max_pooled, global_state], dim=-1)  # (batch, 26)
        return self.net(critic_in)


class ScorerPolicy(nn.Module):
    """
    Unified Actor-Critic policy module for PPO training.
    """

    def __init__(
        self,
        actor_hidden_dims: Optional[List[int]] = None,
        critic_hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.actor = CandidateScorer(input_dim=16, hidden_dims=actor_hidden_dims)
        self.critic = CriticNetwork(
            candidate_dim=10, global_dim=6, hidden_dims=critic_hidden_dims
        )

    def get_action_distribution(
        self, candidates: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Computes masked categorical action distribution.
        Masked invalid actions are set to -1e9 so their probability is strictly 0.0.
        """
        raw_logits = self.actor(candidates)  # (batch, top_k)
        if raw_logits.ndim == 1:
            raw_logits = raw_logits.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        # Action mask: 1 for valid, 0 for invalid
        bool_mask = action_mask.bool()
        # Create masked logits
        masked_logits = torch.where(
            bool_mask, raw_logits, torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype)
        )

        dist = Categorical(logits=masked_logits)
        return dist, raw_logits

    def get_value(
        self,
        candidates: torch.Tensor,
        action_mask: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.critic(candidates, action_mask, global_state)

    def evaluate_actions(
        self,
        candidates: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes log_prob, masked entropy, and state value for given actions.
        Entropy is computed ONLY over unmasked candidates to prevent NaN / numerical instability.
        """
        dist, _ = self.get_action_distribution(candidates, action_mask)
        log_prob = dist.log_prob(actions)

        # Pure masked entropy calculation:
        # H = - sum_{valid} p * log(p)
        probs = dist.probs  # (batch, top_k)
        bool_mask = action_mask.bool()
        log_probs_safe = torch.log(probs.clamp(min=1e-12))
        masked_plogp = torch.where(bool_mask, probs * log_probs_safe, torch.zeros_like(probs))
        entropy = -masked_plogp.sum(dim=-1)

        values = self.get_value(candidates, action_mask, global_state)
        return values.squeeze(-1), log_prob, entropy

    def act(
        self,
        candidates: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects actions from candidate matrix.
        Returns: (actions, log_probs, state_values)
        """
        dist, _ = self.get_action_distribution(candidates, action_mask)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        values = self.get_value(candidates, action_mask)
        return action, log_prob, values.squeeze(-1)
