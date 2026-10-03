#!/usr/bin/env bash
# install_python_deps.sh — Phase 1, Week 1
# Creates a Python virtual environment and installs all NeuroOS-Lite
# dependencies: core, dev (pytest/mypy/ruff), ML (PyTorch), and analysis
# (matplotlib/scipy/pandas).
#
# Usage:
#   bash scripts/setup/install_python_deps.sh [--venv-dir VENV_DIR] [--no-ml]
#
# Options:
#   --venv-dir DIR   Path for the virtual environment (default: .venv)
#   --no-ml          Skip heavy PyTorch/TensorRT installation

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VENV_DIR=".venv"
INSTALL_ML=true

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --no-ml)
            INSTALL_ML=false
            shift
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Python version check (>= 3.10)
# ---------------------------------------------------------------------------
step "Checking Python version …"
PYTHON_BIN="$(command -v python3 || command -v python)"
PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION##*.}"

if [[ "${PYTHON_MAJOR}" -lt 3 ]] || { [[ "${PYTHON_MAJOR}" -eq 3 ]] && [[ "${PYTHON_MINOR}" -lt 10 ]]; }; then
    error "Python >= 3.10 required; found ${PYTHON_VERSION}."
    exit 1
fi
info "Python ${PYTHON_VERSION} ✓"

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
step "Creating virtual environment at '${VENV_DIR}' …"
if [[ -d "${VENV_DIR}" ]]; then
    warn "Virtual environment already exists at '${VENV_DIR}'. Reusing."
else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    info "Virtual environment created ✓"
fi

# Source activation
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate" 2>/dev/null || \
    source "${VENV_DIR}/Scripts/activate" 2>/dev/null || {
        error "Failed to activate virtual environment at '${VENV_DIR}'."
        exit 1
    }

# ---------------------------------------------------------------------------
# 3. Upgrade pip
# ---------------------------------------------------------------------------
step "Upgrading pip …"
pip install --quiet --upgrade pip
info "pip upgraded ✓"

# ---------------------------------------------------------------------------
# 4. Install core + dev dependencies
# ---------------------------------------------------------------------------
step "Installing core and dev dependencies …"
pip install --quiet ".[dev,analysis]"
info "Core + dev + analysis dependencies installed ✓"

# ---------------------------------------------------------------------------
# 5. (Optional) Install ML dependencies
# ---------------------------------------------------------------------------
if "${INSTALL_ML}"; then
    step "Installing ML dependencies (PyTorch + torchvision) …"
    # Try CUDA 12.1 wheel first; fall back to CPU-only if CUDA is unavailable
    if command -v nvcc &>/dev/null || [[ -d /usr/local/cuda ]]; then
        pip install --quiet "torch>=2.1.0" "torchvision>=0.16.0" \
            --index-url https://download.pytorch.org/whl/cu121
        info "PyTorch (CUDA 12.1) installed ✓"
    else
        warn "CUDA not detected. Installing CPU-only PyTorch."
        pip install --quiet "torch>=2.1.0" "torchvision>=0.16.0" \
            --index-url https://download.pytorch.org/whl/cpu
        info "PyTorch (CPU-only) installed ✓"
    fi
else
    warn "--no-ml flag set. Skipping PyTorch installation."
fi

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
step "Verifying installation …"
python -c "import numpy; import pytest; print('numpy', numpy.__version__, '✓')"

if "${INSTALL_ML}"; then
    python -c "import torch; print('torch', torch.__version__, '✓')" 2>/dev/null || \
        warn "PyTorch import verification failed (may need manual inspection)."
fi

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo
info "=== Python environment setup COMPLETE ==="
info "Virtual environment: ${VENV_DIR}/"
info "Python:              $(python --version)"
info ""
info "To activate: source ${VENV_DIR}/bin/activate  (Linux/macOS)"
info "              ${VENV_DIR}\\Scripts\\activate      (Windows)"
info ""
info "To run unit tests: pytest tests/unit -v"
info "To run all tests:  pytest tests/ -v"
