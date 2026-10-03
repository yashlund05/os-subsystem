# Environment Setup Scripts

This directory houses environment initialization scripts for NeuroOS-Lite.

## Status: `IMPLEMENTED` (Completed Phase 1, Week 1)

## Scripts

| Script | Description |
|---|---|
| `bootstrap_linux_kernel.sh` | Verifies Linux kernel ≥ 6.12, `sched_ext` BPF headers, libbpf, and CPU architecture. |
| `install_python_deps.sh` | Creates Python venv and installs core, dev, analysis, and ML (PyTorch) dependencies. |

## Usage

```bash
# Step 1 – verify kernel / sched_ext headers (Linux only)
bash scripts/setup/bootstrap_linux_kernel.sh

# Step 2 – create venv and install Python dependencies
bash scripts/setup/install_python_deps.sh

# Step 2 (skip heavy ML deps)
bash scripts/setup/install_python_deps.sh --no-ml

# Step 2 (custom venv location)
bash scripts/setup/install_python_deps.sh --venv-dir /opt/neuroos-venv
```

After setup, activate the environment and run tests:

```bash
source .venv/bin/activate
pytest tests/unit -v
pytest tests/ -v  # full suite (unit + integration + performance)
```
