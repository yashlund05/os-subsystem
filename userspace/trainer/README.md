# User-Space DRL Training Daemon

This directory contains the off-path deep reinforcement learning training engine executing on a local discrete GPU.

## Status: `NOT IMPLEMENTED` (Scheduled for Phase 2, Week 3)

## Intended Responsibilities
- Drain batched telemetry tuples ($N=256\text{ to }1024$) from pinned host memory to GPU VRAM.
- Train actor-critic architectures (PPO / SAC / Decision Transformer) in PyTorch with NVIDIA TensorRT acceleration.
- Optimize multi-objective reward function balancing mean turnaround time, tail waiting time ($P_{99}$), and context-switch penalties.
- Never block the kernel fast path; execution is fully asynchronous and off-path.
