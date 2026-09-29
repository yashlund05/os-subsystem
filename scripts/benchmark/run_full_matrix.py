"""Full Phase 4 benchmark matrix CLI (Week 7 + Week 8).

Runs REAL simulator sweeps (no fabricated CSVs) and writes JSON artifacts to
experiments/results/ (gitignored). Validates against docs/Schema.md fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root on sys.path when invoked as scripts/benchmark/run_full_matrix.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.ablations.ablations import run_all_ablations
from benchmarks.memory.sweep import run_memory_matrix
from benchmarks.overhead.measure import measure_quantized_latency_ns
from benchmarks.scheduling.sweep import run_scheduling_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuroOS-Lite Phase 4 matrix")
    parser.add_argument("--tasks", type=int, default=60, help="Tasks per scheduling run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--out", type=str, default="experiments/results/phase4_matrix.json")
    args = parser.parse_args()

    print("[Phase4] Running scheduling matrix (7 x 5 x 10)...")
    sched_results = run_scheduling_matrix(num_tasks=args.tasks, seed=args.seed)
    print(f"[Phase4] Scheduling runs: {len(sched_results)}")

    print("[Phase4] Running memory matrix (5 x 4)...")
    mem_results = run_memory_matrix()
    print(f"[Phase4] Memory runs: {len(mem_results)}")

    abl = {}
    if not args.skip_ablations:
        print("[Phase4] Running ablations (PMU / quantization / power)...")
        abl = run_all_ablations(seed=args.seed)

    # Overhead latency on a deterministic random student (emulation; HW via overhead_bench)
    import numpy as np

    rng = np.random.default_rng(args.seed)
    params = {
        "layer_0_weight": rng.standard_normal((8, 16)) * 0.3,
        "layer_0_bias": np.zeros(8),
        "layer_1_weight": rng.standard_normal((1, 8)) * 0.3,
        "layer_1_bias": np.zeros(1),
    }
    from ml.quantization.quantize import quantize_student

    qpolicy, meta = quantize_student(params, version=1)
    lat = measure_quantized_latency_ns(qpolicy, meta["scales"], num_iters=500)

    payload = {
        "scheduling": [r.to_dict() for r in sched_results],
        "memory": [r.to_dict() for r in mem_results],
        "ablations": abl,
        "overhead_emulation": lat,
        "config": {"tasks": args.tasks, "seed": args.seed},
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Phase4] Wrote {out_p} ({len(sched_results)} sched + {len(mem_results)} mem)")


if __name__ == "__main__":
    main()
