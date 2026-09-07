# Product Requirements Document (PRD)

## Project: NeuroOS-Lite
**Asymmetric Local GPU-Trained Neural Preemption and Memory Partitioning for Low-Latency OS Subsystems**

---

### 1. Problem Statement
In classic operating systems research (SOSP, OSDI, EuroSys, IEEE TC, IEEE TPDS), machine learning-based kernel optimizations encounter a fundamental peer-review rejection known as the **Overhead Paradox**:

> *"The decision latency of a neural model (tens of microseconds to milliseconds) exceeds the entire operational time slice or allocation quantum of the subsystem it seeks to optimize (~0.8 to 2.5 µs context switch, <500 ns memory allocation), turning theoretical queue-theoretic efficiency gains into disastrous net throughput collapses."*

Traditional OS heuristics rely on computationally trivial rules ($O(1)$ to $O(n)$ pointer updates):
- **CPU Scheduling**: First-Come First-Served (FCFS), Shortest Job First (SJF), Shortest Remaining Time First (SRTF), Round Robin (RR), Multi-Level Feedback Queue (MLFQ).
- **Memory Allocation**: Fixed Partitioning (MFT), First-Fit, Best-Fit, Binary Buddy.

While lightweight, these heuristics cannot adapt to non-stationary, multi-phase burst dynamics, leading to convoy delays, starvation, cache-trashing preemption loops, and severe external/internal fragmentation under heavy-tailed burst distributions.

---

### 2. Motivation & Core Thesis
NeuroOS-Lite resolves the Overhead Paradox via an **Asymmetric Decoupled Architecture**:
1. **Off-Path Local GPU Synthesis**: Deep reinforcement learning (PPO/SAC) and phase-prediction models execute asynchronously on a local discrete GPU, ingesting high-frequency hardware PMU telemetry across zero-copy lock-free circular memory rings.
2. **Fast-Path In-Kernel Micro-Inference**: Learned policies are distilled into quantized fixed-point integer models (16 $\to$ 8 $\to$ 1 MLP) or lookup tables (LUTs) evaluated within a strict budget of $< 50\text{ ns}$ (target: $\le 45\text{ ns}$) directly inside the OS dispatch path (Linux `sched_ext`).
3. **Dual-Tier Subsystem Optimization**: Unifies preemptive uniprocessor scheduling (dynamic quantum sizing and starvation mitigation) with dynamic memory partitioning (lifetime affinity clustering to suppress external fragmentation).

---

### 3. Research Objectives
- **O1**: Formulate uniprocessor burst-scheduling and dynamic memory partitioning as a constrained Markov Decision Process (MDP).
- **O2**: Engineer an in-kernel quantized micro-inference core achieving sub-50ns deterministic decision latency without floating-point register usage.
- **O3**: Construct a zero-copy lock-free single-producer single-consumer (SPSC) telemetry pipeline streaming 16-byte hardware PMU counter deltas.
- **O4**: Provide bounded, fail-safe execution through an automated $O(1)$ guardrail engine that falls back to classical heuristics if model drift exceeds $3\sigma$ or queue depth saturates ($N > 1024$).
- **O5**: Rigorously benchmark against 5 CPU scheduling and 4 memory allocation baselines across synthetic Pareto bursts, Google Borg cluster traces, and SPEC CPU2017/MiBench workloads.

---

### 4. Target Stakeholders
- **Operating Systems Researchers & Kernel Engineers**: Developing next-generation extensible kernel schedulers (e.g., Linux `sched_ext`).
- **Systems-for-AI / AI-for-Systems Researchers**: Exploring edge deployment of learned system heuristics without throughput degradation.
- **Cloud & Real-Time Subsystem Architects**: Demanding bounded tail latency ($P_{99}, P_{99.9}$) and minimized memory fragmentation.

---

### 5. Functional Requirements
- **FR1: Telemetry Streaming**: Stream 16-byte packed telemetry structs (`task_telemetry`) per context switch across a lock-free SPSC ring buffer without memory allocation in the fast path.
- **FR2: Off-Path DRL Training**: Continuously train uniprocessor policy models using actor-critic RL (PPO/SAC) on local discrete GPU using batched telemetry.
- **FR3: Knowledge Distillation & Quantization**: Distill PyTorch models into 8-bit quantized integer weights and biases, updating shared kernel policy tables atomically.
- **FR4: In-Kernel Micro-Inference**: Evaluate the 16 $\to$ 8 $\to$ 1 integer MLP in $< 50\text{ ns}$ using integer SIMD instructions (AVX2/NEON) without invoking floating-point units.
- **FR5: Dynamic Quantum Computation**: Compute optimal task selection and task-specific execution quanta $\Delta t_q \in [q_{min}, q_{max}]$.
- **FR6: Lifetime-Affinity Memory Binning**: Allocate dynamic heap memory by predicting deallocation horizons $\tau_k$, clustering correlated lifetimes into contiguous memory zones.
- **FR7: Deterministic Guardrail Fallback**: Detect queue saturation ($N > 1024$) or prediction distribution drift ($> 3\sigma$) and revert to classical $O(1)$ scheduling (MLFQ/Red-Black tree) and Buddy allocation instantly.

---

### 6. Non-Functional Requirements
- **NFR1: Dispatch Latency Bound**: Fast-path micro-inference must execute in $< 50\text{ ns}$ (research design target: $\le 45\text{ ns}$).
- **NFR2: Zero Floating-Point in Kernel**: Fast-path execution must use strictly fixed-point / integer SIMD operations to prevent FPU register state saving overhead.
- **NFR3: Non-Blocking Decoupling**: The kernel scheduling tick must NEVER block waiting for GPU training or ring buffer synchronization.
- **NFR4: Memory Footprint**: Kernel micro-core policy weights and states must occupy $< 4\text{ KB}$ of cacheable kernel memory.
- **NFR5: Robustness**: Total makespan and safety must never degrade below classical baselines under adversarial or pathological inputs.

---

### 7. Scope & Explicit Non-Goals

#### In-Scope:
- Preemptive uniprocessor CPU scheduling under varying load factors ($\rho \in [0.10, 0.98]$).
- Dynamic heap memory partitioning and external/internal fragmentation mitigation.
- Cycle-accurate uniprocessor simulation and Linux `sched_ext` kernel integration.
- Offline and local asynchronous GPU reinforcement learning pipeline.

#### Explicit Non-Goals:
- **No Distributed / Multi-Socket NUMA Scheduling**: Multi-socket scheduling and NUMA memory affinity are out of scope for the Lite uniprocessor prototype.
- **No In-Kernel Backpropagation**: Training is strictly off-path on discrete GPUs; no gradient calculations will ever occur in kernel space.
- **No Arbitrary Neural Architectures in Kernel**: LSTMs, Transformers, and deep ResNets are explicitly rejected in the fast path due to microsecond-scale execution costs.
- **No FPU Register Spilling**: Floating-point kernel operations are strictly prohibited.
