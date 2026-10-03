# 10-Week Implementation Roadmap & Phases

This roadmap tracks the development lifecycle of the NeuroOS-Lite research project.

---

### Status Legend
- `[COMPLETED / FOUNDATION]`: Architecture, foundation, common interfaces, build systems, and scaffolding established.
- `[PLANNED]`: Scheduled for the upcoming implementation phase; active focus.
- `[NOT STARTED]`: Detailed in specification; awaiting prerequisite phases.
- `[FUTURE]`: Long-term post-initial-paper exploration.

---

### Phase 1: Foundations & Classical Baselines (Weeks 1–2)

#### Week 1: Scaffolding, Telemetry Interfaces & Workload Traces
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Configure isolated development environment with Linux 6.12+ and `sched_ext` headers.
  - Implement synthetic burst generation tools (Pareto $\alpha \in [1.1, 1.8]$, Poisson $\rho \in [0.10, 0.98]$, Uniform).
  - Implement parser/ingester for Google Borg cluster trace slice and SPEC CPU2017 profiles.
  - Implement memory-mapped lock-free circular SPSC ring buffer for 16-byte `task_telemetry` structs.

#### Week 2: Classical Subsystem Baselines (Simulator Core)
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Implement cycle-accurate uniprocessor CPU scheduling baselines:
    - Non-preemptive: FCFS, SJF.
    - Preemptive: SRTF, Round Robin ($q \in [1\text{ ms}, 50\text{ ms}]$ parametric sweep), standard MLFQ.
  - Implement memory partitioning baselines:
    - Fixed Partitioning (MFT), Dynamic Best-Fit, Dynamic First-Fit, Binary Buddy Allocator.
  - Construct automated evaluation harness calculating TAT, NTAT, Waiting Time, Context Switches, and External Fragmentation.

---

### Phase 2: AI Optimization & Quantized Distillation (Weeks 3–4)

#### Week 3: Asynchronous Local GPU Training Pipeline
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Build PyTorch DRL actor-critic (PPO/SAC) pipeline with TensorRT acceleration.
  - Implement state-space encoder: Task burst history, CPU cycles consumed, IPC, memory access strides, cache misses.
  - Implement multi-objective reward function penalizing turnaround time, tail waiting time ($P_{99}$), and context switches.

#### Week 4: Model Distillation & Quantized In-Kernel Inference
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Perform knowledge distillation: compress deep network into a 2-layer quantized MLP ($16 \to 8 \to 1$).
  - Construct Piecewise Linear Model (PLM) and lookup tables (LUTs) for address offsets and quantum scalers.
  - Implement evaluation engine in pure C using integer SIMD intrinsics (AVX2 / ARM NEON) with zero floating-point operations.
  - Validate cycle execution time on physical hardware via `rdtsc_ordered` against the $\le 45\text{ ns}$ target.

---

### Phase 3: Subsystem Integration & Guardrails (Weeks 5–6)

#### Week 5: Subsystem Dispatch Integration & Guardrails
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Integrate quantized inference core into Linux `sched_ext` kernel dispatch path.
  - Implement dynamic quantum adjustment engine $\Delta t_q \in [q_{min}, q_{max}]$.
  - Implement $O(1)$ fail-safe guardrails (queue depth $N > 1024$ and prediction error $> 3.0\sigma$) with instantaneous fallback to Red-Black tree / MLFQ.

#### Week 6: Learned Memory Partitioning Engine
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Deploy lifetime-prediction clustering for dynamic heap allocation requests.
  - Intercept allocation requests in userspace harness or custom kernel slab allocator.
  - Colocate allocations with correlated deallocation horizons $\tau_k$ into contiguous memory bands.
  - Validate external fragmentation reduction against Best-Fit.

---

### Phase 4: Rigorous Benchmarking & Ablation Analysis (Weeks 7–8)

#### Week 7: Rigorous Benchmarking & Parameter Sweeps
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Execute full benchmark matrix:
    - 7 scheduling algorithms $\times$ 5 workload profiles $\times$ 10 load factors ($\rho \in [0.1, 0.98]$).
    - 4 memory allocators $\times$ 4 synthetic churn traces.
  - Collect empirical distributions for Turnaround Time, Waiting Time ($P_{95}, P_{99}, P_{99.9}$), Context Switches, and External Fragmentation slivers.

#### Week 8: Ablation Studies & Overhead Verification
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Conduct ablation studies:
    - Feature importance: Impact of disabling hardware PMU inputs (running on raw burst history alone).
    - Quantization penalty: Compare 32-bit float accuracy vs. 8-bit integer inference accuracy.
  - Profile end-to-end power consumption via NVIDIA NVML and Intel RAPL counters.

---

### Phase 5: Publication Drafting & Artifact Packaging (Weeks 9–10)

#### Week 9: Paper Writing & IEEE Formatting
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Draft IEEE conference paper (Sections I through V) in LaTeX.
  - Generate publication-grade vector graphics (PDF/SVG) for CDFs, heatmaps, and Pareto trade-off frontiers.
  - Document mathematical proofs and MDP formulation.
- **Deliverables** (2026-09-30):
  - `paper/neuroos_lite.tex`: Full IEEE conference paper in LaTeX (5 sections: Introduction, Related Work, System Design, Implementation, Evaluation, Conclusion).
  - `paper/references.bib`: 14 BibTeX citations (Tsafrir 2007, Arpaci 2018, Wilson 1995, Mao 2019, Jacob 2018, Han 2016, sched_ext 2024, Gymnasium 2023, etc.).
  - `paper/figures/fig1_cdf_waiting_time.{pdf,png}`: CDF of Mean Waiting Time across all 10 policies (Pareto ρ=0.8, 30 seeds).
  - `paper/figures/fig2_pareto_frontier.{pdf,png}`: Pareto trade-off frontier (Decision Overhead ns vs. Mean WT µs).
  - `paper/figures/fig3_memory_heatmap.{pdf,png}`: Dynamic memory allocation heatmap (Best-Fit fragmentation vs. Lifetime-Affinity clustering).
  - `paper/figures/fig4_policy_bar_chart.{pdf,png}`: Grouped bar chart, 5 key policies × 8 workloads.
  - `paper/figures/fig5_learning_curves.{pdf,png}`: BC+PPO Student learning curves across 3 seeds.
  - `paper/figures/fig6_quantization_degradation.{pdf,png}`: Int8 degradation summary with CI bands.
  - `scripts/reproduce/generate_figures.py`: Automated figure generation script (matplotlib + scipy).
  - `paper/Makefile`: LaTeX build automation (pdflatex + bibtex cycle).

#### Week 10: Peer-Review Polish, Artifact Packaging & Rebuttal Prep
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Address key reviewer counterarguments (inference overhead, GPU memory transfer latency, burst estimation reliability, starvation, stability).
  - Package code, trace datasets, and scripts into reproducible Docker / QEMU test harness.
  - Perform artifact evaluation dry runs.
- **Deliverables** (2026-09-30):
  - `paper/REVIEWER_REBUTTAL.md`: Pre-drafted responses to 7 anticipated reviewer counterarguments (hardware generalisability, quantization FLAGGED condition, student > teacher, OOD, ring buffer backpressure, related work scope, BC vs PPO).
  - `scripts/reproduce/package_artifacts.py`: Automated ZIP artifact packager with SHA-256 manifest, REPRODUCE.md, and Docker instructions.
  - All 58 unit tests passing (100% pass rate, 10.3 s).
  - Phase 5 closed.

---

### Phase 6: Multi-Core SMP, NUMA Affinity & Online Drift Self-Correction
- **Status**: `[COMPLETED]`
- **Objectives**:
  - Implement multi-core SMP scheduling extension for Linux `sched_ext` with per-CPU runqueues, cache warmth tracking, and NUMA-aware task placement.
  - Implement hierarchical work-stealing load balancer (intra-NUMA domain search before remote NUMA sockets) with migration dampening.
  - Implement NUMA-aware lifetime affinity memory allocator partitioning heap into per-node lifetime bands to preserve memory bus interconnect bandwidth.
  - Implement in-kernel and userspace online drift self-correction without GPU round-trips, eliminating unnecessary MLFQ fallback trips under sudden burst shifts.
  - Construct multi-core discrete-event simulation engine (`simulator/scheduling/smp_engine.py`) and empirical scaling benchmark suite (`benchmarks/smp/run_smp_benchmark.py`).
- **Deliverables** (2026-10-03):
  - `kernel/include/numa_topology.h`: C NUMA distance matrix, CPU mask, and cache warmth state definitions.
  - `kernel/sched_ext/smp_scheduler.{h,c}`: Core in-kernel SMP scheduling, CPU selection, work stealing, and candidate evaluation.
  - `kernel/inference/online_corrector.{h,c}`: In-kernel integer online drift tracker and residual bias compensator.
  - `simulator/scheduling/smp_engine.py`: Discrete-event SMP simulation engine with NUMA topology and migration cost accounting.
  - `schedulers/smp_neuroos/smp_scheduler.py`: Multi-core learned scheduler mirror with NUMA distance weighting and online drift correction.
  - `allocators/numa_lifetime/numa_allocator.py`: Multi-socket NUMA-aware lifetime affinity allocator.
  - `userspace/drift/online_corrector.py`: Real-time online error tracking and residual bias compensator.
  - `tests/unit/test_phase6_smp.py`: Multi-core scaling and work stealing unit tests (100% pass).
  - `tests/unit/test_phase6_numa_allocator.py`: NUMA-aware allocation and fallback unit tests (100% pass).
  - `tests/unit/test_phase6_online_drift.py`: Online drift adaptation and guardrail stabilization unit tests (100% pass).
  - `benchmarks/smp/run_smp_benchmark.py`: Empirical SMP scaling (1 to 8 cores), NUMA locality (100% local hit rate), and drift suppression benchmark harness.
  - All Phase 6 tests passing, benchmark results recorded in `benchmarks/smp/results.json`.
  - Phase 6 closed.

