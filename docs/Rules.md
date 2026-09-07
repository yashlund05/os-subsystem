# Engineering Rules & Guidelines

## NeuroOS-Lite Research & Engineering Code of Conduct

These rules are mandatory for all human contributors, researchers, and automated AI coding assistants working in this repository.

---

### Rule 1: Do Not Invent Requirements
All implementations must align strictly with the project specification (`docs/PRD.md`, `docs/TRD.md`, and `docs/Phases.md`). If a requirement is underspecified or ambiguous, document it as an open question or propose an explicit RFC rather than making undocumented architectural deviations.

### Rule 2: Absolute Ban on Fabricated Experimental Results
- **Never fabricate results**: Under no circumstances should synthetic benchmark CSVs, fake log files, mock performance numbers, or artificial figures be generated and passed off as real empirical results.
- **Specification Values are Hypotheses**: Numerical values cited from the research specification (e.g., $\le 45\text{ ns}$ inference, 28.4% turnaround reduction, 41.2% $P_{99}$ latency reduction, 38.9% fragmentation reduction) are research targets to be evaluated experimentally in Phase 4. Treat them strictly as claims requiring empirical validation.

### Rule 3: Do Not Silently Change the Architecture
The decoupled asymmetric paradigm (off-path GPU training vs. in-kernel integer micro-inference) is a non-negotiable architectural invariant. Do not move training into the kernel, do not make dispatch block on GPU communication, and do not introduce monolithic neural architectures (e.g., Transformers, LSTMs) into the fast path.

### Rule 4: Keep Kernel Fast-Path Code Deterministic
Every instruction on the scheduling dispatch path and memory allocation path must have bounded worst-case execution time:
- No unbounded while loops.
- No dynamic heap allocations (`kmalloc`/`malloc`) in the fast path.
- No locking or synchronization primitives that could yield or sleep.

### Rule 5: Zero Floating-Point Operations in the Kernel Fast Path
The operating system kernel disables or penalizes floating-point register saves on context switches. The kernel fast-path micro-inference engine must operate strictly using signed integer arithmetic (`int8_t`, `int16_t`, `int32_t`) or integer SIMD intrinsics (AVX2 `pmaddubsw`, ARM NEON `vdot.s8`). Any PR introducing `float` or `double` into kernel code will be rejected.

### Rule 6: Keep GPU Training Strictly Off-Path
Kernel dispatching and task execution must proceed uninterrupted regardless of GPU state:
- If the GPU worker crashes, stalls, or disconnects, the kernel dispatcher continues running with the latest valid policy or falls back to classical heuristics.
- Zero copy overhead: Telemetry ingestion operates exclusively over lock-free single-producer single-consumer (SPSC) circular rings.

### Rule 7: Maintain a Deterministic Fail-Safe Fallback
All learned components must maintain an $O(1)$ fail-safe mechanism:
- If ready queue depth exceeds $N = 1024$, bypass the neural policy.
- If running prediction drift exceeds $3.0\sigma$, trip the fallback guardrail.
- Revert instantaneously to deterministic classical algorithms (MLFQ, SRTF, or Buddy Allocator).

### Rule 8: Benchmark Every Performance-Sensitive Change
Any modification to the ring buffer, fast-path inference core, dispatch hooks, or memory partition search must be benchmarked using cycle-accurate hardware counters (`rdtsc_ordered` or Linux `perf`) on an isolated hardware core. Document the exact compiler version, optimization flags, and CPU pinning setup.

### Rule 9: Separate Experimental Code from Production-Intended Kernel Code
- Python prototyping, exploratory RL training scripts, and simulation harnesses belong in `simulator/`, `ml/`, and `experiments/`.
- C-based kernel code intended for `sched_ext` belongs in `kernel/`.
- Do not mix prototype Python dependencies into the kernel build system.

### Rule 10: Document Assumptions Explicitly
State all hardware platform assumptions (e.g., AVX2 support, TSC invariance, Linux kernel version $\ge 6.12$) in code comments and configuration files.

### Rule 11: Maintain Strict Repository Hygiene
Never commit:
- Trace files (Google Borg cluster traces, SPEC CPU2017 raw traces).
- Neural network weight binaries (`.pt`, `.pth`, `.onnx`, `.bin`).
- Generated experiment output logs, plots, or parquet datasets.
- Secrets, credentials, or `.env` files.
Refer to `ml/datasets/README.md` and `.gitignore`.
