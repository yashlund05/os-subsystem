# NeuroOS-Lite Team Activity & Work Log

This document serves as the single source of truth for ongoing engineering progress, active assignments, architectural decisions, and handoff notes for the 4-member research/engineering team.

---

## Team Roster & Ownership Areas

| Member | Primary Focus Area | Secondary Responsibilities |
|---|---|---|
| **Member 1 (Lead / Systems)** | OS Kernel (`sched_ext`, in-kernel SIMD inference, C micro-core) | Telemetry ring buffer, TSC benchmarking |
| **Member 2 (Simulation)** | Discrete-event simulator engine, classical baselines, workloads | Benchmark runner harness, metric validation |
| **Member 3 (ML / RL)** | GPU DRL training pipeline (PPO/SAC), state space modeling | Feature engineering, reward function design |
| **Member 4 (Quantization)** | Knowledge distillation, int8 quantization, LUT generation | Guardrail drift detection, memory affinity binning |

---

## Chronological Work Log

### [2026-09-28] — Phase 3 Implementation & Audit: Local GPU DRL Training Pipeline & Honest Baseline Analysis
- **Status**: `COMPLETED & AUDITED`
- **Contributors**: ML & Simulation Team (Member 3, Member 4, Pair AI Assistant)
- **Completed Work & Empirical Audit Findings**:
  - **1. MLFQ vs RR Analysis**:
    - Discovered that in `evaluate.py`, both `RR-5ms` and `MLFQ` mapped to `action = 0` unconditionally because MLFQ priority level demotion was not being tracked across steps in `SchedulerEnv`.
    - Fixed: Added priority demotion on quantum expiration (`task.priority_level = min(3, priority_level + 1)`) in `SchedulerEnv` and updated `evaluate.py` to pick candidates with lowest priority level. Verified on hand-built multi-tier workload: MLFQ achieves $17,679.3\,\mu\text{s}$ Mean WT vs. $19,346.7\,\mu\text{s}$ for RR-5ms.
  - **2. Heuristic vs PPO Teacher Comparison (Honest Assessment)**:
    - The observation-space heuristic `argmin(pred_burst - 0.2*age)` **outperformed the PPO Teacher across all workloads and loads**:
      - Pareto $\rho=0.8$: Heuristic Mean WT $433.6 \pm 145.1\,\mu\text{s}$ vs. Teacher $642.5 \pm 202.5\,\mu\text{s}$ (Teacher is $+208.9\,\mu\text{s}$ worse).
      - Pareto $\rho=0.95$: Heuristic $537.7 \pm 165.7\,\mu\text{s}$ vs. Teacher $829.7 \pm 254.0\,\mu\text{s}$ (Teacher is $+292.1\,\mu\text{s}$ worse).
      - Convoy: Heuristic $2,477.5\,\mu\text{s}$ vs. Teacher $7,424.5\,\mu\text{s}$ (Teacher is $+4,947.0\,\mu\text{s}$ worse).
    - *Honest Conclusion*: PPO does **not** add value over a simple 1-line heuristic on this state space.
  - **3. Convoy Workload Breakdown**:
    - Tracing decisions revealed why Teacher equals RR ($7,424.5\,\mu\text{s}$): Task 1 (50ms) arrives at $t=0$ when the ready queue is empty. PPO dispatches Task 1. Because the environment does not interrupt mid-slice without an external timer interrupt or arrival preemption action, and the network output for Task 1 is non-negative, Task 1 runs for its full 5ms quantum before short tasks get dispatched.
  - **4. Reward Alignment Diagnosis**:
    - Discovered that cumulative reward actually **ranks Teacher above Heuristic** ($88.21$ vs $85.21$) despite Heuristic having lower waiting time ($78.3\,\mu\text{s}$ vs $72.1\,\mu\text{s}$). The reward penalty for context switches ($w_{\text{switch}} = 0.20$) heavily disincentivizes preemption, teaching PPO to avoid switching away from long tasks.
  - **5. Student Capacity vs Training**:
    - Trained $16 \to 8 \to 1$ Student for 500k steps: Mean WT reached $1,185.6\,\mu\text{s}$ (did not overtake Heuristic).
    - Supervised regression fit of $16 \to 8 \to 1$ directly on the heuristic rule achieved $R^2 = 0.9637$ and $\text{MSE} = 0.0031$.
    - *Correction*: The 8-neuron student has plenty of representational capacity for the optimal rule ($R^2 > 0.96$). Its failure in RL is an **optimization/exploration failure** under RL reward dynamics, not a network capacity ceiling.
  - **6. Noise Sweep Bugfix & Verification**:
    - Found and fixed a bug where `env.burst_estimator.noise_std` was set instead of `noise_std_frac`. Tested real noise sensitivity: Noise=0 gives $331.2\,\mu\text{s}$, Default (0.1) gives $328.7\,\mu\text{s}$, High (0.5) gives $354.0\,\mu\text{s}$ ($+25.3\,\mu\text{s}$ degradation under severe noise).
  - **7. Extended 500k Teacher Training**:
    - Extended training for 500,000 steps converged in return ($66.2$) but did not improve evaluation Mean WT ($1,085.9\,\mu\text{s}$ vs $642.5\,\mu\text{s}$ at 50k steps). PPO overfits to the context-switch penalty and does not overtake the heuristic.
  - **8. Permutation Feature Importance**:
    - Measured exact delta Mean WT when scrambling each feature on 15 eval seeds:
      - Dominant feature: `burst_ratio` ($+358.3\,\mu\text{s}$ degradation when permuted).
      - Context features: `cpu_busy_frac` ($+56.3\,\mu\text{s}$), `load_factor_norm` ($+52.9\,\mu\text{s}$), `is_running_val` ($+51.4\,\mu\text{s}$).
      - PMU features: `cache_miss_norm` ($+40.8\,\mu\text{s}$), `branch_mispred_norm` ($+40.4\,\mu\text{s}$).
      - Inactive features: `priority_norm` ($+0.0\,\mu\text{s}$), `max_wait_norm` ($-0.2\,\mu\text{s}$).
- **Handoff & Next Steps for Quantization Lead (Member 4)**:
  - **Export Artifacts**: Available under `ml/checkpoints/` (`neuroos_teacher_staged_weights.npz`, `neuroos_student_staged_weights.npz`).
  - **Distillation Strategy**: Because direct RL on student is sub-optimal but student can represent the heuristic with $R^2 = 0.9637$, distillation in Phase 4 should train the student to match either Teacher logits or Heuristic scores directly.

---

### [2026-09-28] — Phase 2B Implementation: Gymnasium CPU Scheduling Environment
- **Status**: `COMPLETED`
- **Contributors**: ML & Simulation Team (Member 3, Member 2, Pair AI Assistant)
- **Completed Work**:
  - Implemented `BurstEstimator` (`userspace/trainer/burst_estimator.py`) modeling runtime kernel burst prediction ($\alpha$-filter + Gaussian noise) without accessing oracle ground-truth bursts.
  - Implemented `ObservationEncoder` (`userspace/trainer/observation.py`) mapping ready-queue candidates to exactly `NEUROOS_INPUT_DIM = 16` per-candidate features (10 task-specific + 6 global context features), directly compatible with the in-kernel micro-core.
  - Implemented `RewardCalculator` (`userspace/trainer/reward.py`) providing multi-objective, independently toggleable terms (step wait penalty normalized by queue size, max wait starvation penalty, context switch penalty, completion bonus, tail threshold proxy).
  - Implemented `SchedulerEnv` (`userspace/trainer/env.py`) adhering to Gymnasium API standards (`reset`, `step`, `action_masks`), with discrete task selection ($K=16$) and graceful fallback safety handling.
  - Implemented `ClassicalSchedulerWrapper` (`userspace/trainer/wrapper.py`) allowing FCFS, SJF, SRTF, RoundRobin, and MLFQ to run inside the Gymnasium environment for direct apples-to-apples comparison.
  - Implemented `make_scheduler_env` (`ml/training/env_factory.py`) with disjoint seed ranges for training (`1000..49999`) and evaluation (`50000..99999`).
  - Unit test suite: added 8 new unit tests in `tests/unit/test_scheduler_env.py` (total 30 tests in repo, 100% passing).
  - Code coverage maintained at 91% across 1,273 statements.
  - Ruff check and formatting clean across all 100 repository files.
- **Handoff & Next Steps for Team**:
  - **Member 3 (ML Lead)**: Can directly instantiate `SchedulerEnv` or `make_scheduler_env("train")` with Stable-Baselines3 / CleanRL `MaskablePPO` using `env.action_masks()` to train the continuous and discrete policy models.
  - **Member 4 (Quantization Lead)**: The 16-D per-candidate feature vector produced by `ObservationEncoder` is directly formatted for the planned 16 $\to$ 8 $\to$ 1 student distillation model.
  - **Member 1 (Kernel Lead)**: The discrete action dispatch semantics and PMU feature bindings align with `kernel/include/neuroos_kernel.h`.

---

### [2026-09-22] — Phase 1 Implementation: Foundations & Classical Baselines
- **Status**: `COMPLETED`
- **Contributors**: Simulation & Systems Team (Member 2, Member 1, Pair AI Assistant)
- **Completed Work**:
  - Initialized `WORK_LOG.md` for team coordination across all 4 project members.
  - Implemented unified `BaseScheduler` abstraction and 5 classical CPU schedulers:
    - `schedulers/fcfs/scheduler.py`: $O(1)$ non-preemptive FIFO queue.
    - `schedulers/sjf/scheduler.py`: $O(\log n)$ non-preemptive min-heap sorted by total burst.
    - `schedulers/srtf/scheduler.py`: $O(\log n)$ preemptive min-heap with dynamic arrival preemption checks.
    - `schedulers/round_robin/scheduler.py`: $O(1)$ circular queue with parametric quantum ($q \in [1\text{ ms}, 50\text{ ms}]$).
    - `schedulers/mlfq/scheduler.py`: $K$-tier feedback queues with geometric quantum scaling ($q_k = q_0 \cdot 2^k$), demotion on quantum expiry, and periodic global starvation boost.
  - Implemented unified `BaseAllocator` abstraction and 4 dynamic memory allocators:
    - `allocators/fixed_partition/allocator.py`: MFT static partition table with internal fragmentation tracking.
    - `allocators/first_fit/allocator.py`: Dynamic variable partition linear scan with block splitting and coalescing.
    - `allocators/best_fit/allocator.py`: Dynamic variable partition smallest-viable-block search with external fragmentation sliver tracking.
    - `allocators/buddy/allocator.py`: Binary Buddy system with power-of-two recursive splitting and XOR buddy coalescing.
  - Implemented cycle-accurate discrete-event simulation engine (`simulator/scheduling/engine.py`) with context-switch penalties ($\bar{t}_{\text{ctx\_save}}$) and metric recording (TAT, NTAT, WT, RT, $P_{95}, P_{99}, P_{99.9}$, overhead ratio $\Phi_{overhead}$).
  - Implemented workload synthesis engine (`simulator/workloads/`):
    - `synthetic.py`: Heavy-tailed Pareto bursts ($\alpha \in [1.1, 1.8]$) and Poisson arrivals across offered load factors ($\rho \in [0.10, 0.98]$).
    - `trace_parser.py`: Google Borg cluster trace parser and SPEC CPU2017 replay model.
    - `adversarial.py`: Pathological convoy triggers and odd-even memory fragmentation churn triggers.
  - Implemented automated benchmark harness (`benchmarks/runner.py`) outputting structured JSON conforming to `docs/Schema.md`.
  - Built comprehensive unit test suite in `tests/unit/`:
    - 22/22 tests passing with 90% overall code coverage.
    - Zero lint errors with `ruff`.
- **Handoff & Next Steps for Team**:
  - **Member 1 (Kernel Lead)**: Reference `kernel/include/` and `schedulers/base.py` to prepare the pure C SPSC ring buffer implementation and sched_ext stub.
  - **Member 2 (Simulation Lead)**: Expand benchmark sweep scripts in `scripts/benchmark/` to generate baseline comparison matrices across load factors $\rho \in [0.10, 0.98]$.
  - **Member 3 (ML Lead)**: Begin Phase 2 (Week 3) offline DRL training pipeline (`userspace/trainer/`) using the simulated gym-like environment hooked into `simulator/scheduling/engine.py`.
  - **Member 4 (Quantization Lead)**: Review `ml/models/policy_interface.py` to prepare the 16 $\to$ 8 $\to$ 1 integer quantization pipeline and lookup table (LUT) builders.

---

### [2026-09-07] — Repository Foundation & Scaffolding
- **Status**: `COMPLETED`
- **Completed Work**:
  - Established full directory structure (82 required paths verified).
  - Authored authoritative documentation suite: `PRD.md`, `TRD.md`, `Rules.md`, `Architecture.md`, `Flow.md`, `Schema.md`, `Phases.md`, `Metrics.md`, `Experimental-Protocol.md`, `IMPLEMENTATION_STATUS.md`.
  - Configured project build systems: `pyproject.toml`, `CMakeLists.txt`, `Makefile`.
  - Configured GitHub Actions CI workflows: `ci.yml` (multi-Python matrix) and `build.yml` (multi-OS C/C++ matrix).
  - Scaffolded hardware C interfaces: `telemetry_event.h` (16-byte packed struct), `neuroos_kernel.h`, `scheduler_interface.h`, `allocator_interface.h`, `ring_buffer.h`.
  - Scaffolded initial Python simulator and passed 5/5 unit tests.
