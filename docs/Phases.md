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
- **Status**: `[PLANNED]` (Repository Foundation Complete)
- **Objectives**:
  - Configure isolated development environment with Linux 6.12+ and `sched_ext` headers.
  - Implement synthetic burst generation tools (Pareto $\alpha \in [1.1, 1.8]$, Poisson $\rho \in [0.10, 0.98]$, Uniform).
  - Implement parser/ingester for Google Borg cluster trace slice and SPEC CPU2017 profiles.
  - Implement memory-mapped lock-free circular SPSC ring buffer for 16-byte `task_telemetry` structs.

#### Week 2: Classical Subsystem Baselines (Simulator Core)
- **Status**: `[NOT STARTED]`
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
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Build PyTorch DRL actor-critic (PPO/SAC) pipeline with TensorRT acceleration.
  - Implement state-space encoder: Task burst history, CPU cycles consumed, IPC, memory access strides, cache misses.
  - Implement multi-objective reward function penalizing turnaround time, tail waiting time ($P_{99}$), and context switches.

#### Week 4: Model Distillation & Quantized In-Kernel Inference
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Perform knowledge distillation: compress deep network into a 2-layer quantized MLP ($16 \to 8 \to 1$).
  - Construct Piecewise Linear Model (PLM) and lookup tables (LUTs) for address offsets and quantum scalers.
  - Implement evaluation engine in pure C using integer SIMD intrinsics (AVX2 / ARM NEON) with zero floating-point operations.
  - Validate cycle execution time on physical hardware via `rdtsc_ordered` against the $\le 45\text{ ns}$ target.

---

### Phase 3: Subsystem Integration & Guardrails (Weeks 5–6)

#### Week 5: Subsystem Dispatch Integration & Guardrails
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Integrate quantized inference core into Linux `sched_ext` kernel dispatch path.
  - Implement dynamic quantum adjustment engine $\Delta t_q \in [q_{min}, q_{max}]$.
  - Implement $O(1)$ fail-safe guardrails (queue depth $N > 1024$ and prediction error $> 3.0\sigma$) with instantaneous fallback to Red-Black tree / MLFQ.

#### Week 6: Learned Memory Partitioning Engine
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Deploy lifetime-prediction clustering for dynamic heap allocation requests.
  - Intercept allocation requests in userspace harness or custom kernel slab allocator.
  - Colocate allocations with correlated deallocation horizons $\tau_k$ into contiguous memory bands.
  - Validate external fragmentation reduction against Best-Fit.

---

### Phase 4: Rigorous Benchmarking & Ablation Analysis (Weeks 7–8)

#### Week 7: Rigorous Benchmarking & Parameter Sweeps
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Execute full benchmark matrix:
    - 7 scheduling algorithms $\times$ 5 workload profiles $\times$ 10 load factors ($\rho \in [0.1, 0.98]$).
    - 4 memory allocators $\times$ 4 synthetic churn traces.
  - Collect empirical distributions for Turnaround Time, Waiting Time ($P_{95}, P_{99}, P_{99.9}$), Context Switches, and External Fragmentation slivers.

#### Week 8: Ablation Studies & Overhead Verification
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Conduct ablation studies:
    - Feature importance: Impact of disabling hardware PMU inputs (running on raw burst history alone).
    - Quantization penalty: Compare 32-bit float accuracy vs. 8-bit integer inference accuracy.
  - Profile end-to-end power consumption via NVIDIA NVML and Intel RAPL counters.

---

### Phase 5: Publication Drafting & Artifact Packaging (Weeks 9–10)

#### Week 9: Paper Writing & IEEE Formatting
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Draft IEEE conference paper (Sections I through V) in LaTeX.
  - Generate publication-grade vector graphics (PDF/SVG) for CDFs, heatmaps, and Pareto trade-off frontiers.
  - Document mathematical proofs and MDP formulation.

#### Week 10: Peer-Review Polish, Artifact Packaging & Rebuttal Prep
- **Status**: `[NOT STARTED]`
- **Objectives**:
  - Address key reviewer counterarguments (inference overhead, GPU memory transfer latency, burst estimation reliability, starvation, stability).
  - Package code, trace datasets, and scripts into reproducible Docker / QEMU test harness.
  - Perform artifact evaluation dry runs.

---

### Phase 6: Future Extensions (Post-Paper)
- **Status**: `[FUTURE]`
- Multi-core SMP uniprocessor migration and NUMA node affinity.
- Online drift self-correction without GPU round-trips.
