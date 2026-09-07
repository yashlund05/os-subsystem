# NeuroOS-Lite

**Asymmetric Local GPU-Trained Neural Preemption and Memory Partitioning for Low-Latency OS Subsystems**

[![CI](https://github.com/your-org/neuroos-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/neuroos-lite/actions/workflows/ci.yml)
[![Build](https://github.com/your-org/neuroos-lite/actions/workflows/build.yml/badge.svg)](https://github.com/your-org/neuroos-lite/actions/workflows/build.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status: Foundation](https://img.shields.io/badge/Status-Foundation%20%2F%20Scaffolding-orange.svg)](docs/IMPLEMENTATION_STATUS.md)

---

## 1. Problem Statement & Motivation

Operating system scheduling and dynamic memory allocation rely on static heuristics (e.g., Round Robin, Shortest Remaining Time First, Best-Fit, Buddy allocators). While computationally negligible ($O(1)$ to $O(n)$ pointer updates), these heuristics fail to capture non-stationary, multi-phase burst dynamics, inducing preemption thrashing, prolonged tail waiting times, and severe external fragmentation under bursty, heavy-tailed workloads.

Conversely, applying deep machine learning inside OS kernels historically faces the **Overhead Paradox**:
> *Standard deep learning and reinforcement learning models introduce tens to hundreds of microseconds of inference latency—dwarfing the 0.8–2.5 µs uniprocessor context switch and <500 ns memory allocation time slices they seek to optimize.*

**NeuroOS-Lite** resolves this trade-off through an **Asymmetric Decoupled Architecture**:
1. **Off-Path GPU Training**: Heavy deep reinforcement learning (PPO / SAC) runs asynchronously on a local discrete GPU, ingesting high-frequency hardware PMU and scheduler telemetry across zero-copy lock-free circular ring buffers.
2. **Fast-Path Micro-Inference**: Learned policies are distilled into quantized fixed-point integer representations (a 16 $\to$ 8 $\to$ 1 MLP or lookup tables) evaluated within a strict budget of $< 50\text{ ns}$ (target: $\le 45\text{ ns}$) directly inside the kernel dispatch path (e.g., Linux `sched_ext`).
3. **Dual-Tier Subsystem Optimization**: Addresses preemptive uniprocessor scheduling (dynamic quantum sizing and anti-starvation) and dynamic memory partitioning (lifetime-affinity clustering to suppress external fragmentation).
4. **Deterministic Guardrails**: Bounded $O(1)$ fail-safe fallback to classical heuristics (MLFQ, Red-Black trees, Buddy allocators) if ready queues saturate ($N > 1024$) or model predictions drift ($> 3\sigma$).

---

## 2. Architecture Overview

```text
+-----------------------------------------------------------------------------------+ 
|                            OFF-PATH: LOCAL GPU (Trainer)                          | 
|  - Telemetry Ingestion (Shared Lockless Circular Ring-Buffer via eBPF / Perf)     | 
|  - Actor-Critic / Continuous Action Space Optimizer (PPO / SAC / Decision Transf.)| 
|  - Quantization-Aware Distillation -> Compressed Integer Lookup / Tiny MLP        | 
+----------------------------------------+------------------------------------------+ 
                                         | Async Policy Update (~1-5 Hz) 
                                         v 
+-----------------------------------------------------------------------------------+ 
|                        FAST-PATH: OS KERNEL CORE (Dispatcher)                     | 
|  - Hardware PMU / Runqueue Telemetry Cache                                        | 
|  - Sub-50ns Quantized Inference Engine (SIMD / Fixed-Point Register Ops)          | 
|  - Decision: Dynamic Quantum Delta t_q, Preemption Trigger, Allocation Bin        | 
|  - Fallback Safe-Path: O(1) Red-Black Runqueue / Buddy Allocator if drift > thr   | 
+-----------------------------------------------------------------------------------+ 
```

Detailed architectural contracts, data flows, and schemas are documented in [`docs/Architecture.md`](docs/Architecture.md), [`docs/Flow.md`](docs/Flow.md), and [`docs/Schema.md`](docs/Schema.md).

---

## 3. Repository Structure

```text
neuroos-lite/
├── README.md                   # Repository overview & setup guide
├── LICENSE                     # Apache 2.0 License
├── CONTRIBUTING.md              # Contributor guidelines
├── .gitignore                  # Data/model/artifact hygiene rules
├── .env.example                # Local environment template
├── Makefile                    # Developer build & test targets
├── pyproject.toml              # Python project configuration (PEP 621)
├── CMakeLists.txt              # C/C++ cross-platform build system
│
├── docs/                       # Authoritative Project Documentation
│   ├── PRD.md                  # Product Requirements Document
│   ├── TRD.md                  # Technical Requirements Document
│   ├── Rules.md                # Engineering & research rules
│   ├── Architecture.md         # System boundaries (off-path vs. fast-path)
│   ├── Flow.md                 # End-to-end dataflow lifecycles
│   ├── Schema.md               # Telemetry, task, policy, decision schemas
│   ├── Phases.md               # 10-week implementation roadmap
│   ├── Metrics.md              # Formal evaluation metric formulas
│   ├── Experimental-Protocol.md# Benchmarking rig & evaluation methodology
│   └── IMPLEMENTATION_STATUS.md# Current status registry
│
├── kernel/                     # Kernel-Level Subsystems (Linux sched_ext)
│   ├── sched_ext/              # Extensible scheduler hooks
│   ├── inference/              # Sub-50ns integer micro-inference engine
│   ├── telemetry/              # In-kernel PMU counter collection
│   ├── guardrails/             # O(1) fail-safe drift detection & fallback
│   └── include/                # Kernel C header definitions
│
├── userspace/                  # User-Space Daemon & Model Training
│   ├── trainer/                # Local GPU DRL training pipeline (PPO/SAC)
│   ├── distillation/           # Quantization-aware distillation engine
│   ├── policy/                 # Double-buffered atomic policy table
│   └── telemetry/              # SPSC ring buffer drain worker
│
├── simulator/                  # Cycle-Accurate Research Simulator
│   ├── scheduling/             # Discrete-event uniprocessor CPU queue
│   ├── memory/                 # Dynamic memory heap simulator
│   ├── workloads/              # Synthetic & trace workload generators
│   └── README.md
│
├── schedulers/                 # Scheduling Algorithm Implementations
│   ├── include/                # Unified scheduler abstraction interface
│   ├── fcfs/                   # First-Come First-Served
│   ├── sjf/                    # Shortest Job First
│   ├── srtf/                   # Shortest Remaining Time First
│   ├── round_robin/            # Parametric Round Robin (q = 5ms, 20ms)
│   └── mlfq/                   # Multi-Level Feedback Queue
│
├── allocators/                 # Memory Allocator Implementations
│   ├── include/                # Unified allocator abstraction interface
│   ├── fixed_partition/        # Multiprogramming with Fixed Tasks (MFT)
│   ├── first_fit/              # First-Fit Dynamic Variable Partitioning
│   ├── best_fit/               # Best-Fit Dynamic Variable Partitioning
│   └── buddy/                  # Binary Buddy Allocator
│
├── telemetry/                  # Telemetry Subsystem
│   ├── ring_buffer/            # Lock-free SPSC circular ring buffer
│   ├── ebpf/                   # Linux eBPF tracepoints
│   ├── pmu/                    # Hardware PMU counters
│   └── schemas/                # Event structures and serialization
│
├── ml/                         # Machine Learning Pipeline
│   ├── datasets/               # Trace ingestion & dataset hygiene
│   ├── preprocessing/          # Normalization & rolling window aggregators
│   ├── features/               # PMU & execution phase feature encoders
│   ├── training/               # Off-path RL policy models
│   ├── distillation/           # Student model quantization (16->8->1)
│   ├── quantization/           # Integer LUT & fixed-point SIMD kernels
│   ├── evaluation/             # Policy loss & reward evaluation
│   └── models/                 # Model definitions
│
├── benchmarks/                 # Micro-Benchmarking & Evaluation
│   ├── scheduling/             # TAT, Waiting Time, Context Switch sweeps
│   ├── memory/                 # External & internal fragmentation sweeps
│   ├── overhead/               # In-kernel cycle timing (rdtsc_ordered)
│   ├── ablations/              # Feature & quantization ablations
│   └── workloads/              # Pareto, Poisson, and trace workload suites
│
├── experiments/                # Experiment Configs & Visualizations
│   ├── configs/                # Matrix parameter configs (YAML/JSON)
│   ├── runners/                # Automated experiment runners
│   ├── results/                # Raw experimental data (gitignored)
│   └── plots/                  # Publication figures (gitignored)
│
├── tests/                      # Automated Testing Suite
│   ├── unit/                   # Python & C unit tests
│   ├── integration/            # Cross-boundary ring buffer & queue tests
│   ├── kernel/                 # Header & struct alignment tests
│   └── performance/            # Sub-50ns latency regression tests
│
├── scripts/                    # Utility & Automation Scripts
│   ├── setup/                  # Environment & dependency bootstrapping
│   ├── build/                  # Build scripts
│   ├── benchmark/              # Benchmark invocation scripts
│   └── reproduce/              # End-to-end paper reproduction scripts
│
├── configs/                    # Subsystem Configuration Templates
│   ├── scheduler/              # Quantum budgets & queue thresholds
│   ├── allocator/              # Heap sizing & bin bounds
│   ├── model/                  # MLP topology & distillation hyperparams
│   └── experiments/            # Benchmark sweep matrix definitions
│
└── .github/workflows/          # Continuous Integration
    ├── ci.yml                  # Linting, Python tests & schema validation
    └── build.yml               # C/C++ cross-platform build validation
```

---

## 4. Current Implementation Status

> [!NOTE]
> The repository is currently in the **FOUNDATION / SCAFFOLDING STAGE**.
>
> Core architectural specifications, C interface headers, Python simulator skeletons, build infrastructure, and CI workflows are established. Active algorithms and deep learning training loops will be implemented milestone-by-milestone per [`docs/Phases.md`](docs/Phases.md).

See [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) for detailed per-component status.

---

## 5. Development Prerequisites & Setup

### Prerequisites
- **Python**: 3.10 or later
- **C/C++ Compiler**: C11 and C++17 compliant (GCC 11+, Clang 13+, or MSVC 2022)
- **Build Tools**: CMake 3.20+, GNU Make
- **Target OS (Kernel Deployment)**: Linux Kernel $\ge 6.12$ with `sched_ext` enabled (`CONFIG_BPF_SCHED=y`)
- **GPU (Off-Path Training)**: NVIDIA GPU with CUDA 12.0+ and TensorRT (optional for simulation)

### Quickstart

```bash
# Clone the repository
git clone https://github.com/your-org/neuroos-lite.git
cd neuroos-lite

# Setup Python environment
make setup

# Build C/C++ scaffolded components
make build

# Run unit tests and header sanity checks
make test

# Run code style & static analysis linters
make lint
```

---

## 6. Testing & Benchmarking

### Automated Unit Tests
```bash
# Run Python unit tests
pytest tests/unit -v

# Run C header alignment & struct packing validation
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Benchmarks (Placeholder)
```bash
make benchmark
```
*Note: In this Foundation stage, `make benchmark` reports notice that Phase 4 benchmarks are scheduled per `docs/Phases.md` and `docs/Experimental-Protocol.md`.*

---

## 7. Research Reproducibility Philosophy & Target Notice

> [!IMPORTANT]
> **Research Targets vs. Empirical Measurements**:
> The numerical performance figures referenced in the initial IEEE research specification:
> - $\le 45\text{ ns}$ micro-inference latency (sub-50ns bound)
> - Up to 28.4% mean turnaround time reduction
> - 41.2% $P_{99}$ waiting time reduction
> - 34.6% context switch reduction
> - 38.9% external fragmentation reduction
>
> are **design targets and research hypotheses to be empirically tested** in Phase 4. They are NOT claimed as completed results of this foundation repository. No artificial benchmarks, mock CSVs, or synthetic figures are committed to this repository (see [`docs/Rules.md`](docs/Rules.md)).

---

## 8. 10-Week Research Roadmap

- **Phase 1 (Weeks 1–2)**: Scaffolding, Telemetry Ring, Workload Traces & Classical Baselines
- **Phase 2 (Weeks 3–4)**: Asynchronous GPU DRL Pipeline, Model Distillation & In-Kernel Fixed-Point SIMD
- **Phase 3 (Weeks 5–6)**: Linux `sched_ext` Dispatch Integration, Dynamic Quantum Sizing & Guardrails
- **Phase 4 (Weeks 7–8)**: Rigorous Benchmarking Sweeps, Ablations & RAPL/NVML Power Profiling
- **Phase 5 (Weeks 9–10)**: IEEE Conference Paper Drafting, Artifact Packaging & Rebuttal Prep

Detailed milestone tracking is maintained in [`docs/Phases.md`](docs/Phases.md).

---

## 9. License & Contributing

- Distributed under the [Apache-2.0 License](LICENSE).
- Contribution guidelines and code of conduct are detailed in [CONTRIBUTING.md](CONTRIBUTING.md) and [`docs/Rules.md`](docs/Rules.md).
