"""Phase 4 experiment runner (experiments/ entry point, wraps scripts/benchmark)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch Phase 4 experiments")
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/results/phase4_matrix.json")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "scripts/benchmark/run_full_matrix.py",
        "--tasks",
        str(args.tasks),
        "--seed",
        str(args.seed),
        "--out",
        args.out,
    ]
    print("[run_phase4] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
