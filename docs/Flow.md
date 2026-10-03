# End-to-End System Flow

## Telemetry Collection, Policy Generation, and Fast-Path Decision Flow

This document outlines the detailed execution lifecycles and data flows across the NeuroOS-Lite subsystem.

---

### 1. High-Level End-to-End Flow Diagram

```text
[Task Context Switch]
        │
        ▼
[Kernel PMU / State Hook] ──> Format 16-byte `task_telemetry`
        │
        ├──> Push to Lock-Free SPSC Ring Buffer (Wait-free, Kernel Fast-Path)
        │
        ▼ (Off-Path Boundary)
[Ring Buffer Drain Daemon] (User-Space, POSIX Shared Memory / mmap)
        │
        ▼
[Batch Aggregator] (Accumulates N=256-1024 samples)
        │
        ▼
[GPU DRL Trainer] (PyTorch / TensorRT / CUDA)
        │  - Compute Loss: L_sched = alpha*TAT + beta*max(WT) + gamma*CtxSwitches
        │  - Optimize Actor-Critic weights (PPO / SAC)
        │
        ▼
[Distillation Engine]
        │  - Compress Deep Teacher Network -> 16 -> 8 -> 1 Student MLP
        │  - 8-bit Integer Quantization (int8 weights, int16 bias)
        │  - Build Piecewise Linear Model (PLM) / LUT Tables
        │
        ▼
[Atomic Policy Updater] (Async ~1-5 Hz)
        │  - Double-buffered pointer exchange to Shared Memory
        │
        ▼ (Fast-Path Boundary)
[Kernel Dispatcher Tick (sched_ext)]
        │
        ├──> 1. Read Ready Queue Telemetry Cache
        ├──> 2. Read Active Quantized Policy Table
        ├──> 3. Perform Branchless Fixed-Point Micro-Inference (< 50 ns)
        ├──> 4. Evaluate Guardrail Engine:
        │         ├── Condition A: Queue Depth > 1024?
        │         └── Condition B: Running Prediction Drift > 3.0 sigma?
        │
        ├──> If Guardrail Trips:
        │         └── Dispatch via Classical O(1) MLFQ / SRTF Fallback
        │
        └──> If Guardrail OK:
                  └── Execute NeuroOS-Lite Decision (Task T*, Dynamic Quantum delta_t_q)
```

---

### 2. Telemetry Ingestion Flow (Kernel -> User Space)

1. **Context-Switch Trigger**:
   - The Linux `sched_ext` hook or simulator event engine fires on task preemption, yield, or arrival.
2. **PMU Delta Read**:
   - Read performance counter registers: elapsed cycles/us, L1/LLC cache miss delta, branch misprediction delta, memory footprint.
3. **Struct Assembly**:
   - Populate `struct task_telemetry` (16 bytes).
4. **Ring Buffer Insertion**:
   - Atomic store to `ring_buffer->entries[head & mask]`.
   - Update `head` pointer with release memory barrier.
   - If buffer is full, increment `dropped_telemetry_counter` without blocking the dispatcher.

---

### 3. Asynchronous Policy Synthesis Flow (User-Space GPU)

1. **Ring Buffer Dequeue**:
   - User-space daemon polls `ring_buffer` with acquire semantics, copying batches into pinned host memory.
2. **GPU Ingestion**:
   - Async DMA transfer (CUDA stream) batches telemetry tuples into GPU tensors.
3. **Reward & Loss Computation**:
   - Computes state representations:
     $$\mathbf{s}_i(t) = \left[ c_i(t), \hat{b}_i(t), w_i(t), \mu_{cache}(i), \beta_{br}(i) \right]^T$$
   - Evaluates multi-objective rewards penalizing turnaround time, starvation, and context switches.
4. **Distillation & Quantization**:
   - Teacher network passes soft targets to the 2-layer $16 \to 8 \to 1$ student network.
   - Quantization-Aware Training (QAT) clamps weights into $[-128, 127]$.
5. **Atomic Swap**:
   - Distilled integer weights are serialized into `struct neuroos_quantized_policy`.
   - Update version tag and swap active pointer in shared memory.

---

### 4. Fast-Path Scheduling Dispatch Flow

1. **Dispatch Invocation**:
   - Kernel calls `ops.dispatch` to select next runnable thread.
2. **Feature Vector Assembly**:
   - The kernel extracts 16 normalized integer features from the ready queue tasks.
3. **Micro-Inference ($< 50\text{ ns}$)**:
   - Branchless dot-product using AVX2 integer instructions (`pmaddubsw` / `paddb`).
   - Computes task score and dynamic quantum budget $\Delta t_q \in [q_{min}, q_{max}]$.
4. **Guardrail Check**:
   - Evaluates queue depth $N$ against `NEUROOS_MAX_QUEUE_DEPTH` (1024).
   - Evaluates empirical error against `NEUROOS_DRIFT_THRESHOLD` ($3.0\sigma$).
   - If invalid: bypass AI decision and select task via $O(1)$ Red-Black tree / MLFQ.
5. **Execution**:
   - Enqueue selected task to CPU execution hardware with budget $\Delta t_q$.
