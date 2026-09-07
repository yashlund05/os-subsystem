# Technical Requirements Document (TRD)

## Project: NeuroOS-Lite
**Asymmetric Local GPU-Trained Neural Preemption and Memory Partitioning for Low-Latency OS Subsystems**

---

### 1. System Overview and Asymmetric Decoupling
NeuroOS-Lite decomposes kernel subsystem optimization into two decoupled, asynchronous execution realms:
1. **The Fast-Path Dispatch Realm (OS Kernel Space)**:
   - Evaluates preemption and allocation decisions synchronously on every scheduling tick and allocation call.
   - Operates under a strict deterministic budget: $< 50\text{ ns}$ (target: $\le 45\text{ ns}$).
   - Disallows all floating-point math, thread preemption, and memory allocation.
2. **The Off-Path Optimization Realm (User-Space GPU Daemon)**:
   - Ingests high-frequency hardware PMU and scheduling telemetry asynchronously.
   - Trains deep reinforcement learning (RL) policies using actor-critic algorithms (PPO / SAC).
   - Distills converged models into quantized integer weights and compresses them into lookup structures.
   - Synchronizes policy weights via atomic memory pointers at a low frequency (~1 to 5 Hz).

```text
+---------------------------------------------------------------------------------+
|                               SYSTEM BUS & REGISTERS                            |
|                                                                                 |
|   +---------------------+    Ring Buffer (mmap)     +-----------------------+   |
|   | OS Kernel Telemetry | ------------------------> | Local GPU Worker      |   |
|   |  - eBPF sched_ext   |                           |  - PyTorch DRL Engine |   |
|   |  - PMU Counters     | <------------------------ |  - Model Distiller    |   |
|   +---------------------+    Atomic Shared Table    +-----------------------+   |
|              |                                                                  |
|              v                                                                  |
|   +-------------------------------------------------------------------------+   |
|   | In-Kernel Fast Path (Sub-50ns)                                          |   |
|   |  - Fixed-Point SIMD Quantized Dot-Product                               |   |
|   |  - Lookup-Table (LUT) Interpolation                                     |   |
|   |  - Deterministic Fallback: Revert to SRTF/Buddy if drift detected       |   |
|   +-------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------+
```

---

### 2. Telemetry Ingestion Pipeline & SPSC Ring Buffer
- **Mechanism**: Single-Producer Single-Consumer (SPSC) circular ring buffer mapped into user space via `remap_pfn_range` (or standard memory-mapped POSIX shared memory in simulator).
- **Structure Size**: Strictly 16 bytes per context-switch event to minimize bus traffic and cache thrashing.
  ```c
  struct __attribute__((packed)) task_telemetry {
      uint32_t pid;
      uint16_t elapsed_us;
      uint16_t cache_misses_delta;
      uint16_t branch_mispred_delta;
      uint16_t mem_footprint_kb;
      uint16_t flags;
  };
  ```
- **Hardware Performance Monitoring Unit (PMU)**:
  - Reads retired instruction counters, L1/LLC data cache misses, and branch mispredictions.
  - Telemetry features capture dynamic workload phase transitions far earlier than static exponential averaging ($\alpha$-filtering).
- **Synchronization**: Wait-free producer in the kernel dispatch path. Consumer in user space batches events in chunks of 256–1024 for GPU consumption.

---

### 3. In-Kernel Micro-Inference Core
- **Model Topology**: 2-layer quantized Multi-Layer Perceptron ($16 \to 8 \to 1$).
  - **Inputs (16 fixed-point features)**: Elapsed burst, estimated residual burst, accumulated wait time, cache miss rate, branch misprediction rate, memory footprint, ready queue depth, and normalized historical phase indicators.
  - **Hidden Layer**: 8 neurons with integer ReLU activation: $h_j = \max\left(0, \sum_{i=1}^{16} x_i w_{ij}^{(1)} + b_j^{(1)}\right)$.
  - **Output Layer**: 1 scalar priority score / quantum budget.
- **Quantization Scheme**:
  - Symmetric 8-bit signed integer weights ($w \in [-128, 127]$).
  - 16-bit signed integer hidden activations and bias.
  - 32-bit signed integer accumulation register.
  - Zero floating-point instructions (`no-fpu`).
- **SIMD Acceleration**:
  - x86_64: `_mm256_maddubs_epi16`, `_mm256_add_epi16`, `_mm256_packs_epi16`.
  - ARM64: NEON `vdot_s8`.
- **Target Latency Budget**: Execution verified via `rdtsc_ordered` to stay below 50 ns (< 150 cycles @ 3.0 GHz).

---

### 4. Linux `sched_ext` & Dispatcher Integration
- **Framework**: Linux Kernel 6.12+ `sched_ext` (eBPF-extensible scheduling core).
- **Dispatch Hook**: `ops.select_cpu`, `ops.enqueue`, and `ops.dispatch`.
- **Dynamic Quantum Computation**: The model outputs both task selection and a dynamic time slice $\Delta t_q \in [q_{min}, q_{max}]$ (e.g., $1\text{ ms} \le \Delta t_q \le 50\text{ ms}$).
  - Compute-bound steady tasks receive larger quanta to minimize context-switch overhead.
  - Interactive, memory-bound, or bursty tasks receive tight quanta with aggressive preemption rights.

---

### 5. Dynamic Memory Partitioning Subsystem
- **Heap Representation**: Contiguous memory buffer $M_{total}$ partitioned into variable-sized dynamic blocks $\mathcal{B} = \{B_1, \dots, B_m\}$.
- **Lifetime Affinity Clustering**:
  - Request format: $R_k = \langle S_k, \tau_k \rangle$ where $S_k$ is requested size and $\tau_k$ is the AI-predicted allocation horizon.
  - Allocator colocates allocations with correlated expected release times into adjacent physical memory zones.
  - Minimizes external fragmentation slivers by ensuring contiguous block deallocations occur simultaneously.
- **Search Latency**: Bounded lookup ($O(1)$ or $O(\log N)$) through lifetime affinity indexing.

---

### 6. Fail-Safe Guardrail Engine
To guarantee system stability under unseen adversarial workloads or distribution shift:
1. **Queue Saturation Guardrail**:
   - If ready queue depth $N > 1024$, bypass neural inference and immediately use classical $O(1)$ MLFQ / Red-Black tree dispatch.
2. **Distribution Drift Guardrail**:
   - Running variance counters monitor empirical burst progression versus predicted progression.
   - If prediction error exceeds $3.0\sigma$, trip the guardrail and revert to deterministic fallback.
3. **Memory Allocator Guardrail**:
   - If lifetime-affinity search cannot satisfy an allocation within budget $\epsilon$, fallback instantaneously to standard Buddy Allocator.

---

### 7. Benchmarking Architecture
- **Hardware Testbed Specification**:
  - Host CPU: Intel Core i9-13900K (pinned to isolated P-core, no frequency scaling).
  - GPU: NVIDIA RTX 4090 (24GB VRAM, TensorRT execution).
  - OS: Linux 6.12-rc with `sched_ext`.
- **Timing Harness**:
  ```c
  static inline uint64_t rdtsc_ordered(void) {
      uint32_t lo, hi;
      asm volatile("lfence\n\trdtsc\n\t" : "=a"(lo), "=d"(hi) :: "memory");
      return ((uint64_t)hi << 32) | lo;
  }
  ```
- **Baselines Supported**:
  - Schedulers: FCFS, SJF, SRTF, Round Robin ($q=5\text{ms}, 20\text{ms}$), MLFQ.
  - Allocators: Fixed Partitioning (MFT), First-Fit, Best-Fit, Binary Buddy.
