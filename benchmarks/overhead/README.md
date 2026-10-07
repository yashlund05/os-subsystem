# In-Kernel Overhead & TSC Micro-Benchmarks

This directory contains cycle-accurate micro-benchmarks measuring in-kernel evaluation latency using `rdtsc_ordered` on pinned hardware cores.

## Status: `IMPLEMENTED` (Completed Phase 4, Week 8)
- Target: $\le 45\text{ ns}$ (< 150 cycles @ 3.0 GHz)
- Guardrail: $< 50\text{ ns}$

## Artifact: `results.csv`

Raw benchmark output from `kernel/inference/overhead_bench.c` is committed to
`benchmarks/overhead/results.csv`. This file is the authoritative measurement
artifact cited in **Section V-F** of the paper.

> **Remediation item 0.4**: The `results.csv` file currently contains a
> schema placeholder. To close this item, compile and run `overhead_bench` on
> real hardware (see instructions below) and commit the raw CSV output.

## Build and Run

```bash
# Prerequisites: GCC >= 11 or Clang >= 14, AVX2-capable x86_64 CPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target overhead_bench

# Run pinned to an isolated core to minimise OS jitter
taskset -c 2 ./build/overhead_bench > benchmarks/overhead/results.csv
git add benchmarks/overhead/results.csv
git commit -m "bench: add real overhead_bench results on <machine description>"
```

## CSV Schema

| Column        | Description                                         |
|---------------|-----------------------------------------------------|
| `iters`       | Number of timed iterations (100 000)                |
| `mean_cycles` | Mean TSC cycles per `neuroos_micro_infer_score` call|
| `min_cycles`  | Minimum observed cycles                             |
| `max_cycles`  | Maximum observed cycles                             |
| `sink`        | Accumulator sink (prevents dead-code elimination)   |

## Notes

- Compile with `-mavx2` to exercise the AVX2 fast path added in item 0.1.
- The scalar fallback path is always compiled in; benchmark both if comparing.
- The requantization step (item 0.2) is now included in the measured path.
