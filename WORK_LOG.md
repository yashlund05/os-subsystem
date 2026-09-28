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

### [2026-09-28] — Phase 3 Full Audit & Empirical Resolution: Tasks 1–8 Verified
- **Status**: `AUDITED, RECONCILED & BENCHMARKED (V3 Canonical)`
- **Contributors**: Full Team (Member 1, Member 2, Member 3, Member 4, Pair AI Assistant)
- **Verified Empirical Findings**:
  - **Task 1: Heuristic Validity & Oracle Identity**:
    - At sigma=0, observation heuristic decisions match Heuristic-Oracle exactly (730/730 decisions, 100.00% agreement, 0 diffs).
    - At sigma=0.1, observation heuristic is within 0.01% of oracle metrics (254.8 us vs 254.7 us).
    - Age tuning sweep on TRAIN seeds (1000..1029) determined optimal w_age = 0.5 (257.92 us on train).
  - **Task 2: Noise-Heuristic Mechanism**:
    - Previously, without PID task burst context, default prior 5000us caused all tasks to appear identical, degenerating to FCFS. Rising noise added tie-breaking perturbations that broke convoys.
    - With properly centered task estimation, performance strictly and monotonically degrades with noise: sigma=0.0 (254.7 us) -> sigma=0.1 (254.8 us) -> sigma=0.3 (259.7 us) -> sigma=0.5 (289.6 us).
  - **Task 3: Side-Channel Audit (PMU / Memory Leakage)**:
    - In `simulator/workloads/synthetic.py`, cache_misses (r=0.9841) and branch_mispredictions (r=0.9913) are directly scaled from burst duration, providing massive indirect burst leakage.
    - Teacher ablation on Pareto rho=0.8: Zeroing PMU features degrades Teacher by +785.8 us (1924.1 -> 2709.9 us); Shuffling degrades Teacher by +611.5 us (2535.5 us). Proves policy actively exploits PMU counters.
  - **Task 4: Reward Sensitivity & Little's Law**:
    - Correlation between current step wait term and True Total Waiting Time is only r=0.3889 because queue_len cancels out in (step_elapsed * queue_len) / (5000 * max(1, queue_len)), penalizing makespan rather than queue waiting time.
    - Proposed Little's-law term: -(N_waiting * dt) / T_norm. Integrates mathematically to total waiting time (r=1.0000). Context switch penalty is conceptually redundant if switch time accrues as waiting time.
  - **Task 5: Controlled Retraining & Invalidation**:
    - Retrained Teacher and Student models with new monotonic remaining-time encoder across 3 training seeds (1001, 1002, 1003).
    - Marked all legacy checkpoints as `INVALID_SUPERSEDED`.
  - **Task 6: Supervised Student**:
    - Raw heuristic quantity fit yields R2=0.850. The 16->8->1 MLP with 8 ReLUs approximates the 2D linear plane in 16-D space with minor projection residual. Evaluated on all workloads: Pareto (3168.1 us vs 3075.8 us), Poisson (1269.0 us vs 1349.5 us), Convoy (16421.8 us vs 11310.6 us).
  - **Task 7: SJF Discrepancy & MLFQ Confirmation**:
    - Non-preemptive SJF achieves 2449.3 us (matches canonical 2454 us). Preemptive SRTF achieves 209.8 us (matches original ~228 us). Conflation in earlier notes resolved: SJF is non-preemptive and suffers head-of-line blocking.
    - Verified environment eval path uses real Phase 1 `MLFQScheduler` class with geometric tiers [5000, 10000, 20000].
  - **Task 8: Burst Prediction Error Assumptions**:
    - Literature citations: Tsafrir et al. (IEEE TPDS 2007) reported 35-45% mean runtime estimation error in production systems; Arpaci-Dusseau (OSTEP Ch. 8) details EMA smoothing. sigma=0.30 formulated as documented project baseline assumption.

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
