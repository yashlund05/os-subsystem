# NeuroOS-Lite Top-Level Makefile
# Provides uniform development targets for C/C++ and Python subsystems

.PHONY: all setup build test lint format benchmark clean help

PYTHON ?= python
CMAKE ?= cmake
BUILD_DIR ?= build

all: build

help:
	@echo "NeuroOS-Lite Foundation Build Targets:"
	@echo "  make setup      - Install Python development dependencies"
	@echo "  make build      - Configure and build C/C++ scaffold targets"
	@echo "  make test       - Run Python and C header tests"
	@echo "  make lint       - Run ruff/mypy and style linters"
	@echo "  make format     - Auto-format codebases"
	@echo "  make benchmark  - Placeholder for Phase 4 benchmarking suite"
	@echo "  make clean      - Clean build directories and caches"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

build:
	$(CMAKE) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(BUILD_DIR) --config Release

test:
	@echo "Running Python test suite..."
	$(PYTHON) -m pytest tests/unit -v
	@echo "Running C header interface test..."
	ctest --test-dir $(BUILD_DIR) --output-on-failure || true

lint:
	$(PYTHON) -m ruff check . || true
	$(PYTHON) -m mypy simulator benchmarks ml --ignore-missing-imports || true

format:
	$(PYTHON) -m ruff format . || true

benchmark:
	@echo "======================================================================"
	@echo "[NOTE] Benchmarks are NOT yet implemented in FOUNDATION stage."
	@echo "Full benchmark matrix (7 schedulers x 5 workloads x 10 load factors)"
	@echo "will be executed in Phase 4 per docs/Phases.md and docs/Experimental-Protocol.md."
	@echo "======================================================================"

clean:
	rm -rf $(BUILD_DIR)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
