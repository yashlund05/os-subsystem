#!/usr/bin/env python3
"""Phase 5 — Artifact Packaging Script.

Packages all NeuroOS-Lite research artifacts into a reproducible archive:
  - Source code (Python + C/C++ headers)
  - Model checkpoints (quantized int8 weights + metadata)
  - Benchmark results (canonical JSON tables)
  - Paper (LaTeX source + generated figures)
  - Docker/QEMU instructions

Usage:
    python scripts/reproduce/package_artifacts.py [--output artifacts_v1.zip]

The resulting archive can be submitted alongside the IEEE paper for
artifact evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = ROOT / "artifacts_phase5"


# ── Files and directories to include in the artifact package ─────────────────
INCLUDE_PATTERNS = [
    # Python source
    "schedulers/**/*.py",
    "allocators/**/*.py",
    "simulator/**/*.py",
    "userspace/**/*.py",
    "ml/**/*.py",
    "quantization/**/*.py",
    "telemetry/**/*.py",
    "kernel/**/*.py",
    "benchmarks/**/*.py",
    "experiments/**/*.py",
    "tests/**/*.py",
    # C / C++ kernel headers
    "kernel/include/*.h",
    "kernel/inference/*.c",
    "kernel/inference/*.h",
    "kernel/sched_ext/*.c",
    "kernel/sched_ext/*.h",
    "kernel/guardrails/*.c",
    "kernel/guardrails/*.h",
    "kernel/telemetry/*.c",
    "kernel/telemetry/*.h",
    # Model artifacts
    "ml/checkpoints/quantized_student_int8.json",
    "ml/checkpoints/quantized_student_int8.npz",
    "ml/checkpoints/canonical_phase4_table.md",
    "ml/checkpoints/canonical_phase4_results.json",
    "ml/checkpoints/canonical_v4_table.md",
    "ml/checkpoints/canonical_v4_results.json",
    "ml/checkpoints/bc_ppo_learning_curves.json",
    # Paper
    "paper/neuroos_lite.tex",
    "paper/references.bib",
    "paper/Makefile",
    "paper/figures/*.pdf",
    "paper/figures/*.png",
    # Documentation
    "docs/PRD.md",
    "docs/TRD.md",
    "docs/Architecture.md",
    "docs/Flow.md",
    "docs/Phases.md",
    "docs/Metrics.md",
    "docs/Experimental-Protocol.md",
    "docs/IMPLEMENTATION_STATUS.md",
    "docs/Rules.md",
    "docs/Schema.md",
    # Build & config
    "pyproject.toml",
    "CMakeLists.txt",
    "Makefile",
    "README.md",
    "WORK_LOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    ".env.example",
    # Reproduction scripts
    "scripts/reproduce/*.py",
    "scripts/reproduce/*.sh",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    "*.pt",  # large pytorch checkpoints (not needed for reproducibility)
    "experiments/results",
    "experiments/plots",
]


def should_exclude(path: Path) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if pat.startswith("*"):
            if path.name.endswith(pat[1:]):
                return True
        else:
            if pat in str(path):
                return True
    return False


def collect_files() -> list[Path]:
    """Collect all files matching include patterns."""
    collected = []
    for pattern in INCLUDE_PATTERNS:
        matches = sorted(ROOT.glob(pattern))
        for m in matches:
            if m.is_file() and not should_exclude(m):
                collected.append(m)
    # Deduplicate
    return list(dict.fromkeys(collected))


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(files: list[Path], archive_path: Path) -> dict:
    """Build artifact manifest with file checksums."""
    manifest = {
        "artifact_name": "NeuroOS-Lite Phase 5 Research Artifact",
        "version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "paper": "NeuroOS-Lite: Asymmetric Local GPU-Trained Neural Preemption "
                 "and Memory Partitioning for Low-Latency OS Subsystems",
        "phase": "Phase 5 — Publication Drafting & Artifact Packaging",
        "archive_sha256": "",  # filled after creation
        "files": [],
    }
    for f in files:
        rel = f.relative_to(ROOT)
        manifest["files"].append({
            "path": str(rel).replace("\\", "/"),
            "sha256": sha256_file(f),
            "size_bytes": f.stat().st_size,
        })
    return manifest


def create_archive(output_path: Path, files: list[Path], manifest: dict) -> None:
    """Create a ZIP archive containing all files and the manifest."""
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            rel = f.relative_to(ROOT)
            arc_name = f"neuroos_lite/{str(rel).replace(chr(92), '/')}"
            zf.write(f, arc_name)
        # Write manifest
        zf.writestr(
            "neuroos_lite/ARTIFACT_MANIFEST.json",
            json.dumps(manifest, indent=2),
        )
        # Write reproduction README
        zf.writestr(
            "neuroos_lite/REPRODUCE.md",
            REPRODUCE_README,
        )
    # Update archive checksum in manifest
    archive_sha = sha256_file(output_path)
    manifest["archive_sha256"] = archive_sha
    print(f"  Archive SHA-256: {archive_sha}")


REPRODUCE_README = """\
# NeuroOS-Lite Artifact Reproduction Guide

## Environment Setup

```bash
# 1. Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. (Optional) CUDA 12 for GPU training
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Reproduction: Phase 4 Benchmark Verification

```bash
# Run the canonical Phase 4 benchmark (requires quantized checkpoint)
python ml/training/generate_phase4_canonical_table.py
# Output: ml/checkpoints/canonical_phase4_table.md
```

## Generate Publication Figures

```bash
pip install matplotlib scipy
python scripts/reproduce/generate_figures.py
# Output: paper/figures/*.pdf, paper/figures/*.png
```

## Compile LaTeX Paper

```bash
cd paper
pdflatex neuroos_lite.tex
bibtex neuroos_lite
pdflatex neuroos_lite.tex
pdflatex neuroos_lite.tex
# Output: paper/neuroos_lite.pdf
```

## Build and Test Kernel C Components

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run Full Test Suite

```bash
pytest tests/ -v
```

## Docker Reproduction (Linux only)

```bash
docker build -t neuroos-lite .
docker run --rm neuroos-lite pytest tests/ -v
docker run --rm neuroos-lite python ml/training/generate_phase4_canonical_table.py
```

## Key Files

| File | Purpose |
|---|---|
| `ml/checkpoints/quantized_student_int8.json` | Int8 quantization metadata & scales |
| `ml/checkpoints/quantized_student_int8.npz` | Int8 weight arrays (W1, b1, W2, b2) |
| `kernel/include/neuroos_weights.h` | CRC32-validated C header weight export |
| `ml/checkpoints/canonical_phase4_table.md` | Phase 4 canonical benchmark results |
| `ml/checkpoints/canonical_v4_table.md` | Full 10-policy comparison table |
| `paper/neuroos_lite.tex` | IEEE conference paper LaTeX source |
| `paper/references.bib` | BibTeX references |

## Expected Results

- 58/58 unit tests passing
- Int8 Student: Mean WT ≤ 702.8 µs (Pareto ρ=0.5), ≤ 1238.8 µs (Pareto ρ=0.8)
- Quantization degradation ≤ 5% on 6/8 workloads (Pareto ρ=0.95: 8.48%, CI overlapping)
- Bit-exact Convoy trace (0 mismatches, 7424.5 µs)
"""


def main():
    parser = argparse.ArgumentParser(description="Package NeuroOS-Lite Phase 5 artifacts.")
    parser.add_argument("--output", default="neuroos_lite_phase5_artifact.zip",
                        help="Output ZIP archive path (default: neuroos_lite_phase5_artifact.zip)")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    print("NeuroOS-Lite Phase 5 Artifact Packager")
    print("=" * 50)
    print(f"Root: {ROOT}")
    print(f"Output: {output_path}")
    print()

    print("Collecting files...")
    files = collect_files()
    print(f"  Found {len(files)} files to package.")

    print("Building manifest...")
    manifest = build_manifest(files, output_path)

    print(f"Creating archive: {output_path.name}")
    create_archive(output_path, files, manifest)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[✓] Archive created: {output_path}")
    print(f"    Size: {size_mb:.2f} MB")
    print(f"    Files: {len(files)}")
    print(f"\nTo verify integrity:")
    print(f"  python -c \"import hashlib,pathlib; "
          f"print(hashlib.sha256(pathlib.Path('{output_path}').read_bytes()).hexdigest())\"")


if __name__ == "__main__":
    main()
