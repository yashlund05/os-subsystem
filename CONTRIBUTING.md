# Contributing to NeuroOS-Lite

Thank you for your interest in contributing to **NeuroOS-Lite**.

NeuroOS-Lite is a systems and machine learning research project investigating asymmetric, local GPU-trained neural preemption and memory partitioning for low-latency operating system subsystems.

## Core Research and Engineering Principles

All contributors and automated agents must adhere to the rules defined in [`docs/Rules.md`](docs/Rules.md):

1. **No Invented Requirements**: Adhere strictly to the project specification and roadmap.
2. **No Fabricated Results**: Numerical values reported in the IEEE specification (such as $\le 45\text{ ns}$ inference, 28.4% turnaround time improvement, 41.2% $P_{99}$ waiting-time reduction, 38.9% fragmentation reduction) are research targets and hypotheses requiring empirical validation. Never commit fabricated benchmarks, artificial CSVs, or synthetic performance plots.
3. **Architectural Decoupling**: Keep GPU training completely off the critical dispatch path. The kernel fast-path must never block waiting for GPU completion.
4. **Kernel Fast-Path Determinism**: Fast-path code must be deterministic with bounded execution time ($< 50\text{ ns}$). Absolutely no floating-point operations in the kernel fast path.
5. **Deterministic Fallback**: If variance drift exceeds threshold ($> 3\sigma$) or queue depth saturates ($N > 1024$), the system must fail-safe to classical deterministic schedulers ($O(1)$ Red-Black tree / MLFQ) and allocators (Buddy).
6. **Data Hygiene**: Do not commit large datasets, cluster traces (Google Borg, SPEC CPU2017), or raw weights to the repository. Follow [`ml/datasets/README.md`](ml/datasets/README.md).

## Development Workflow

### Prerequisites
- Python 3.10+
- CMake 3.20+
- C11 / C++17 compliant compiler (GCC 11+, Clang 13+, or MSVC 2022)
- Linux 6.12-rc with `sched_ext` (for kernel-level target deployment)

### Local Setup
```bash
# Clone the repository
git clone https://github.com/your-org/neuroos-lite.git
cd neuroos-lite

# Setup virtual environment and build tools
make setup

# Build scaffolded binaries and libraries
make build

# Run unit and sanity tests
make test

# Check linting and code styles
make lint
```

### Pull Request Guidelines
- Ensure all CI tests pass (`ci.yml` and `build.yml`).
- Include tests for any newly implemented scheduler, allocator, or telemetry component.
- Separate prototype/experimental code from production-intended kernel code.
- Provide documentation updates for any schema or interface alterations.
