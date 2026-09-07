# System Architecture

## Architectural Blueprint & Boundary Specification

NeuroOS-Lite breaks the historical trade-off between expressive machine learning policies and microsecond OS dispatch constraints by establishing an **Asymmetric Decoupled Boundary**.

---

### 1. The Dual-Realm Boundary

The system cleanly separates heavy, non-deterministic learning algorithms from the ultra-low-latency, cycle-bounded kernel dispatch loop:

```text
+=============================================================================+
|                                OFF-PATH REALM                               |
|                         (User-Space Local GPU Daemon)                       |
+=============================================================================+
                               Kernel Telemetry
                                      │
                                      ▼
                            Lock-Free Ring Buffer
                                      │
                                      ▼
                                 GPU Worker
                                      │
                                      ▼
                          DRL / Policy Optimization
                                      │
                                      ▼
                              Model Distillation
                                      │
                                      ▼
                               Quantized Policy
                                      │
                                      ▼
                             Atomic Policy Update
                                      │
                                      │ (~1-5 Hz Async Memory Swap)
                                      ▼
+=============================================================================+
|                               FAST-PATH REALM                               |
|                      (OS Kernel Dispatch Core: sched_ext)                   |
+=============================================================================+
                              Kernel Dispatcher
                                      │
                                      ▼
                               Telemetry Cache
                                      │
                                      ▼
                              Quantized Inference
                                      │
                                      ▼
                   Scheduling / Quantum / Allocation Decision
                                      │
                                      ▼
                             Guardrail Validation
                                      │
                                      ▼
                        Execute or Deterministic Fallback
```

---

### 2. Off-Path Architecture (Local GPU Trainer)

The off-path pipeline runs in user space on a dedicated local discrete GPU (e.g., NVIDIA RTX 4090):

1. **Telemetry Ingestion**:
   - Reads 16-byte `task_telemetry` events streamed from the kernel across a lock-free Single-Producer Single-Consumer (SPSC) circular buffer.
   - Decoupled from kernel scheduling: if the consumer falls behind, oldest non-critical events are overwritten or batched without blocking the kernel.
2. **Actor-Critic Continuous / Discrete Optimization**:
   - Deep Reinforcement Learning (PPO / SAC / Decision Transformer) optimizes a multi-objective cost function:
     $$\mathcal{L}_{sched} = \alpha \cdot \bar{W}_{turnaround} + \beta \cdot \max_i(w_i(t)) + \gamma \cdot N_{ctx\_switches}$$
   - State space incorporates elapsed execution time, remaining burst estimates, cache miss rates (PMU), branch mispredictions, and ready queue pressure.
3. **Knowledge Distillation & Quantization**:
   - Compresses deep networks into a compact 2-layer quantized MLP ($16 \to 8 \to 1$).
   - Quantizes FP32 weights into 8-bit signed integers ($[-128, 127]$).
   - Generates piecewise linear interpolation tables (LUTs) for rapid address offsets and quantum scalers.
4. **Atomic Policy Synchronization**:
   - Writes new weights and biases to shared memory using double-buffered atomic pointer swaps (`rcu`-like pattern).
   - Occurs at low frequency (~1–5 Hz), introducing zero jitter to the high-frequency dispatch loop.

---

### 3. Fast-Path Architecture (OS Kernel Core)

The fast path operates inside the kernel dispatch tick (via Linux `sched_ext` hooks):

1. **Telemetry Cache Access**:
   - Accesses cached PMU counters and task descriptors directly from per-CPU cache lines.
   - Collects 16 fixed-point features in registers.
2. **Quantized SIMD Micro-Inference**:
   - Executes integer dot-product using AVX2 or ARM NEON instructions.
   - Evaluates within a strict latency bound ($< 50\text{ ns}$, design target $\le 45\text{ ns}$).
   - Employs zero floating-point operations, eliminating FPU register-saving penalties during context switches.
3. **Multi-Objective Decision Synthesis**:
   - Selects next task $T^* \in \mathcal{Q}_R$.
   - Computes adaptive quantum $\Delta t_q \in [q_{min}, q_{max}]$.
   - In memory partitioning, predicts lifetime affinity bin $\tau_k$ for contiguous allocation.
4. **Guardrail Validation & Deterministic Fallback**:
   - Validates running variance against prediction drift ($> 3\sigma$).
   - Validates ready queue depth against saturation limit ($N > 1024$).
   - If guardrails pass, executes decision immediately.
   - If guardrails trip, instantly falls back to classical deterministic algorithms:
     - Scheduler fallback: $O(1)$ MLFQ or Red-Black tree (SRTF).
     - Allocator fallback: $O(\log M)$ Binary Buddy Allocator.

---

### 4. Memory Partitioning Architecture

The memory subsystem tackles external fragmentation in variable-partition dynamic memory (MVT):
- Traditional allocators (Best-Fit / First-Fit) suffer from severe fragmentation holes because they disregard object lifetime affinity.
- NeuroOS-Lite classifies allocation requests $R_k = \langle S_k, \tau_k \rangle$ by predicted deallocation horizon $\tau_k$.
- Objects with similar expected destruction horizons are colocated into contiguous memory bands, ensuring that memory blocks are freed together, eliminating fragmentation slivers.
