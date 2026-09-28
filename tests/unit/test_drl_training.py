"""Unit and integration tests for Phase 3 DRL training pipeline."""

import json
from pathlib import Path

import numpy as np
import torch

from ml.training.export import (
    export_policy_for_quantization,
    numpy_scorer_forward,
)
from ml.training.policy import CandidateScorer, CriticNetwork, ScorerPolicy
from ml.training.ppo import PPOConfig, PPOTrainer
from ml.training.vec_env import VectorSchedulerEnv
from userspace.trainer.env import SchedulerEnv


def test_candidate_scorer_forward_shapes():
    """Verify CandidateScorer produces expected tensor dimensions and permutation equivariance."""
    scorer = CandidateScorer(input_dim=16, hidden_dims=[64, 32])
    # Single candidate
    x1 = torch.randn(16)
    out1 = scorer(x1)
    assert out1.shape == ()

    # Top-K candidate matrix
    xk = torch.randn(8, 16)
    outk = scorer(xk)
    assert outk.shape == (8,)

    # Batched matrix
    xb = torch.randn(4, 8, 16)
    outb = scorer(xb)
    assert outb.shape == (4, 8)


def test_policy_action_masking_zero_probability():
    """Verify that invalid candidate actions strictly receive probability 0.0."""
    policy = ScorerPolicy(actor_hidden_dims=[64, 32], critic_hidden_dims=[32, 32])
    candidates = torch.randn(2, 8, 16)
    action_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.int8)

    dist, raw_logits = policy.get_action_distribution(candidates, action_mask)
    probs = dist.probs  # (2, 8)

    # Valid candidates must have positive probability
    assert (probs[0, :2] > 0).all()
    # Invalid candidates must have strictly 0 probability
    assert (probs[0, 2:] == 0.0).all()

    assert (probs[1, :4] > 0).all()
    assert (probs[1, 4:] == 0.0).all()

    # Sum of probabilities must be 1.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_entropy_no_nan_with_masking():
    """Ensure masked entropy computation never generates NaN even with heavily padded slots."""
    policy = ScorerPolicy(actor_hidden_dims=[8], critic_hidden_dims=[16])
    candidates = torch.randn(4, 16, 16)
    action_mask = torch.zeros((4, 16), dtype=torch.int8)
    action_mask[:, 0] = 1  # Only 1 valid action

    actions = torch.zeros(4, dtype=torch.int64)
    values, log_probs, entropy = policy.evaluate_actions(candidates, action_mask, actions)

    assert not torch.isnan(values).any()
    assert not torch.isnan(log_probs).any()
    assert not torch.isnan(entropy).any()
    # With only 1 valid choice, entropy should be 0.0
    assert torch.allclose(entropy, torch.zeros(4), atol=1e-5)


def test_critic_network_decoupled():
    """Verify critic network operates independently and produces scalar value."""
    critic = CriticNetwork(candidate_dim=10, global_dim=6, hidden_dims=[32, 32])
    candidates = torch.randn(3, 8, 16)
    mask = torch.ones((3, 8), dtype=torch.int8)

    val = critic(candidates, mask)
    assert val.shape == (3, 1)


def test_export_numpy_matches_pytorch(tmp_path: Path):
    """Verify that export_policy_for_quantization and numpy_scorer_forward match PyTorch within 1e-5."""
    student_scorer = CandidateScorer(input_dim=16, hidden_dims=[8])
    student_scorer.eval()

    npz_p, json_p = export_policy_for_quantization(student_scorer, str(tmp_path), prefix="test_student")
    assert npz_p.exists()
    assert json_p.exists()

    with open(json_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["input_dim"] == 16
    assert meta["hidden_dims"] == [8]
    assert len(meta["feature_order"]) == 16
    assert meta["feature_slices"]["task_features"]["dim"] == 10
    assert meta["feature_slices"]["global_features"]["dim"] == 6

    # Load weights for NumPy forward pass
    params = dict(np.load(npz_p))
    candidates_np = np.random.randn(8, 16).astype(np.float32)
    mask_np = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)

    # NumPy forward
    np_probs = numpy_scorer_forward(params, candidates_np, mask_np)

    # PyTorch forward
    with torch.no_grad():
        t_cands = torch.from_numpy(candidates_np).unsqueeze(0)
        t_mask = torch.from_numpy(mask_np).unsqueeze(0)
        policy = ScorerPolicy(actor_hidden_dims=[8])
        policy.actor = student_scorer
        dist, _ = policy.get_action_distribution(t_cands, t_mask)
        pt_probs = dist.probs[0].numpy()

    np.testing.assert_allclose(np_probs, pt_probs, atol=1e-5)


def test_checkpoint_save_and_load_identical_outputs(tmp_path: Path):
    """Verify that saving and loading a checkpoint reproduces exact identical forward outputs."""
    policy = ScorerPolicy(actor_hidden_dims=[64, 32], critic_hidden_dims=[32, 32])
    ckpt_path = tmp_path / "model.pt"

    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "actor_hidden_dims": [64, 32],
        },
        ckpt_path,
    )

    loaded_policy = ScorerPolicy(actor_hidden_dims=[64, 32], critic_hidden_dims=[32, 32])
    ckpt = torch.load(ckpt_path)
    loaded_policy.load_state_dict(ckpt["model_state_dict"])
    loaded_policy.eval()
    policy.eval()

    cands = torch.randn(4, 16, 16)
    mask = torch.ones((4, 16), dtype=torch.int8)

    with torch.no_grad():
        a1, lp1, v1 = policy.act(cands, mask, deterministic=True)
        a2, lp2, v2 = loaded_policy.act(cands, mask, deterministic=True)

    assert torch.equal(a1, a2)
    assert torch.allclose(lp1, lp2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_smoke_training_few_hundred_steps():
    """Verify end-to-end PPO rollout and update on VectorSchedulerEnv for small step count."""
    def env_factory():
        return SchedulerEnv(top_k=8, max_steps=50)

    vec_env = VectorSchedulerEnv([env_factory, env_factory])
    policy = ScorerPolicy(actor_hidden_dims=[8], critic_hidden_dims=[16])
    cfg = PPOConfig(num_rollout_steps=16, num_epochs=2, batch_size=16)

    trainer = PPOTrainer(policy=policy, vec_env=vec_env, config=cfg, device=torch.device("cpu"))
    obs_cands, obs_masks = vec_env.reset()

    buffers, _, _, _ = trainer.collect_rollouts(obs_cands, obs_masks)
    metrics = trainer.update(buffers, progress=0.1)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert not np.isnan(metrics["policy_loss"])
    assert not np.isnan(metrics["value_loss"])
    vec_env.close()


def test_train_single_run_and_eval_suite(tmp_path: Path):
    """Integration test verifying train_single_run and run_evaluation_suite execute cleanly."""
    from ml.training.evaluate import run_evaluation_suite
    from ml.training.train import train_single_run

    cfg = {
        "environment": {"num_envs": 2, "num_tasks": 10, "load_factor": 0.8, "top_k": 8},
        "reward": {
            "step_wait_penalty_weight": 1.0,
            "max_wait_penalty_weight": 0.1,
            "context_switch_penalty_weight": 0.05,
            "completion_bonus": 2.0,
            "tail_penalty_weight": 0.5,
        },
        "ppo": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_coef": 0.5,
            "entropy_coef_start": 0.02,
            "entropy_coef_end": 0.001,
            "max_grad_norm": 0.5,
            "num_rollout_steps": 16,
            "num_epochs": 2,
            "batch_size": 16,
        },
        "curriculum": {"total_timesteps": 64},
        "network": {"critic_hidden_dims": [16, 16]},
    }

    policy, summary = train_single_run(
        cfg=cfg,
        train_seed=123,
        device=torch.device("cpu"),
        actor_hidden_dims=[8],
        curriculum_mode="mixed",
        run_name="test_run",
    )
    assert summary["total_timesteps"] >= 32
    assert "wall_clock_seconds" in summary

    ckpt_p = tmp_path / "test_model.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "actor_hidden_dims": [8],
        },
        ckpt_p,
    )

    eval_results = run_evaluation_suite(
        checkpoint_path=str(ckpt_p),
        num_eval_seeds=2,
        eval_seed_base=50000,
        loads=[0.8],
        workload_types=["pareto"],
    )
    assert "pareto" in eval_results
    assert "load_0.8" in eval_results["pareto"]
    assert "PPO-Policy" in eval_results["pareto"]["load_0.8"]
    assert "mean_wt_us" in eval_results["pareto"]["load_0.8"]["PPO-Policy"]


def test_train_cli_smoke():
    """Smoke test running train.py CLI with ppo_quick config and temporary output directory."""
    import sys

    from ml.training.train import main

    test_args = [
        "train.py",
        "--config",
        "configs/ppo_quick.yaml",
        "--device",
        "cpu",
    ]
    orig_argv = sys.argv
    try:
        sys.argv = test_args
        main()
    finally:
        sys.argv = orig_argv


