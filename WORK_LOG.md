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

### [2026-09-28] — Phase 3 Final Reconciled Resolution: Real Per-Process Predictor, Preemption Cue, Multi-Burst Benchmarks, & Canonical Table
- **Status**: `RESOLVED, BENCHMARKED & RECONCILED (Final Canonical)`
- **Contributors**: Full Team (Member 1, Member 2, Member 3, Member 4, Pair AI Assistant)
- **Verified Empirical Findings & Implementations**:
  1. **Unstarted Task Estimate & FCFS Degeneration**:
     - In single-burst workloads where every task has a unique PID, `BurstEstimator` has no history and falls back to `default_estimate_us = 5000` $\pm 10\%$. Because all unstarted candidates receive nearly identical estimates ($\approx 5000\,\mu\text{s}$), the heuristic score $-\hat{B}/100k + \text{age}/500k$ is dominated entirely by arrival age, degenerating into FCFS with random tie-breaking. Re-labeled from $\sigma=0.30$ to `Heuristic-Prior (5ms Default Fallback)`.
  2. **Realistic Per-Process Predictor & Alternating Workload**:
     - Implemented per-PID EMA, last-burst, rolling recent runtime average, sleep time before wakeup, and priority in `BurstEstimator`.
     - Built alternating CPU burst-sleep workload generator (`generate_multiburst_process_workload`).
     - Real predictor error across 30 eval seeds (3,000 bursts): subsequent bursts relative MAE is **0.2543 (25.43%)**, median relative error is **0.2113 (21.13%)**; overall median relative error is **0.2389 (23.89%)**, matching real OS kernel capabilities at wakeup.
  3. **Arrival Preemption Cue & Heavy-Tail Hazard Rate**:
     - Added running task's remaining estimate to global feature 15 (`running_rem_norm`), enabling candidates to directly compute $\Delta_{\text{preempt}} = \max(0.0, x_{15} - x_9)$.
     - Fixed Gambler's Fallacy clamping in `BurstEstimator`: when a task executes past its estimate, expected remaining time updates conditionally as $\max(1000, 0.5 \times \text{elapsed})$, reflecting heavy-tailed decreasing hazard rate.
     - Convoy Mean WT dropped from $13,425.2\,\mu\text{s}$ down to **$8,925.2\,\mu\text{s}$** (Heuristic-Obs) and **$7,424.5\,\mu\text{s}$** (Teacher & Student, matching RR!).
  4. **Multi-Burst Canonical Benchmarks for All 10 Policies**:
     - Full comparative matrix generated for FCFS, SJF, SRTF, RR, MLFQ, Heuristic-Oracle, Heuristic-Obs (Real Predictor), Supervised-Student, Teacher, and Student.
     - Student (BC+PPO) achieved **2,487.5 $\mu$s**, outperforming MLFQ ($3,797.0\,\mu\text{s}$) by 34.5%, RR ($5,081.7\,\mu\text{s}$) by 51.1%, and FCFS ($5,690.9\,\mu\text{s}$) by 56.3% with lowest switches (89.7).
  5. **FCFS/SJF Convoy Reconciliation**:
     - Proven analytically and empirically: in `create_convoy_workload`, the 50ms head arrives at $t=0$ when the CPU is idle. Non-preemptive SJF dispatches it and cannot preempt when 49 short jobs arrive at $t=1..49$. All 49 short jobs wait for the full 50ms duration, giving exact identical mean WT of $51,327.5\,\mu\text{s}$ for both FCFS and SJF. Preemptive SRTF achieves $2,426.5\,\mu\text{s}$.
  6. **100k-Step 3-Seed BC-PPO Runs & Eval Learning Curves**:
     - Evaluated Teacher and Student across seeds 1001, 1002, 1003 on CUDA with fixed eval seeds. Student converged to **$1,238.8\,\mu\text{s}$** on Pareto $\rho=0.8$ and **$702.8\,\mu\text{s}$** on Pareto $\rho=0.5$.

### [2026-09-28] — Phase 3 Reconciled Resolution: Little's Law, Side-Channel Leak Fix, Standardized Student, BC-PPO, & Canonical V4 Benchmark
- **Status**: `AUDITED, LEAKS FIXED, LITTLE'S LAW VERIFIED, V4 CANONICAL BENCHMARKED`
- **Contributors**: Full Team (Member 1, Member 2, Member 3, Member 4, Pair AI Assistant)
- **Verified Empirical Results**:
  1. **Little's Law Reward Function**:
     - Formulated step wait penalty as $-(N_{\text{waiting}} \cdot \Delta t)/T_{\text{norm}}$ ($T_{\text{norm}} = 100,000.0\,\mu\text{s}$, $w_{\text{switch}} = 0.02$, `w_completion = 0.0` dropped).
     - Pearson correlation between cumulative episode step wait penalty and True Total Waiting Time verified at **$r = 1.000000$ exact** (eliminating the previous $r=0.3889$ makespan artifact).
  2. **Side-Channel PMU Leak Fix**:
     - Fixed `cache_misses` and `branch_mispredictions` in `SimulatedTask` to be dynamic `@property` functions of elapsed execution time only (`int(executed_burst_us * rate)`), with rates generated independently of burst duration.
     - Correlation audit over 6,472 unstarted candidate tasks confirmed Pearson $r = 0.0000$ and Spearman $\rho = 0.0000$ with true total burst.
  3. **Multi-Burst Workload & Real EMA Evaluation**:
     - Implemented `generate_multiburst_process_workload` in `simulator/workloads/synthetic.py` with recurring PIDs and per-process Pareto means.
     - Evaluated pure EMA across 30 seeds: first-burst relative MAE is 13.23 (prior fallback); subsequent bursts relative MAE is 1.63 (median relative error 54.52%), providing genuine imperfect estimation. Documented $\sigma=0.30$ strictly as a documented synthetic benchmark assumption.
  4. **Supervised Student Optimization**:
     - Closed-form least squares on raw features proved exact linear recovery ($R^2 = 1.000000$, weights on burst_est and age: $-1.0000, 1.0000$, max residual on other 14 features $1.43 \times 10^{-8}$).
     - Standardized target training of $16 \to 8 \to 1$ neural network reached $R^2 = 0.9989$, with $\sim 96\%$ top-1 action agreement with the observation heuristic in-env (Pareto: 3,222 $\mu$s vs 3,076 $\mu$s; Poisson: 1,241 $\mu$s vs 1,350 $\mu$s; Convoy: 11,704 $\mu$s vs 11,311 $\mu$s).
  5. **Behavior-Cloning Pretrained PPO Fine-Tuning**:
     - Pretrained Teacher ($16 \to 64 \to 32 \to 1$) and Student ($16 \to 8 \to 1$) with BC on heuristic decisions ($R^2 > 0.999$), followed by PPO fine-tuning under Little's Law across 3 seeds (`1001, 1002, 1003`) on CUDA.
     - Teacher achieved 1,319.0 $\mu$s on Pareto $\rho=0.5$ (outperforming observation heuristic 1,983.1 $\mu$s by 33.5%) and 2,247.4 $\mu$s on Pareto $\rho=0.8$ (beating observation heuristic 3,075.8 $\mu$s by 26.9%).
  6. **Canonical V4 Benchmark Results**:
     - Evaluated all 10 policies across all 7 scenarios over the 30 eval seeds (`50000..50029`) with 95% confidence intervals.
     - Full test suite verified: 41/41 tests passing cleanly.

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
