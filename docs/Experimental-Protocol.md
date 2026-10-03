# Experimental Protocol & Verification Rig

## Standardized Empirical Evaluation Protocol

This protocol defines the rigorous testing methodology for benchmarking NeuroOS-Lite against classic heuristic baselines in Phase 4.

> [!IMPORTANT]
> **No Fake Results**: In adherence to `docs/Rules.md`, this protocol defines experimental procedures, workload generators, and measurement rigs. No fabricated data, synthetic CSVs, or unverified claims are included.

---

### 1. Hardware & Software Testbed Rig

To ensure reproducibility across all runs, the experimental harness runs on a standardized hardware and software environment:

```text
+-----------------------------------------------------------------------------------+
|                            HARDWARE/SOFTWARE TESTBED                              |
|  - Host CPU: Intel Core i9-13900K (8P + 16E cores, pinned to isolated P-core)     |
|  - GPU: NVIDIA RTX 4090 (24GB VRAM, TensorRT / FP8 Execution)                     |
|  - OS: Linux Kernel 6.12-rc with sched_ext (eBPF extensible scheduler core)      |
|  - Micro-benchmarking Framework: Google Benchmark / Linux `perf` stat             |
+-----------------------------------------------------------------------------------+
```

#### Deterministic Pinning Configuration:
- CPU Frequency Scaling Governor: set to `performance` (disable Intel SpeedStep / C-states / Turbo boost fluctuations).
- Process Affinity: Scheduler fast path and benchmarks pinned to isolated physical Core 2 (`isolcpus=2 nohz_full=2`).
- Kernel Preemption: Preemptible uniprocessor model with high-resolution timers (`CONFIG_HZ=1000`).

---

### 2. Micro-Benchmark Timing Protocol

Micro-second and nano-second execution intervals are recorded using serializing TSC instructions:

```c
static inline uint64_t rdtsc_ordered(void) {
    uint32_t lo, hi;
    asm volatile("lfence\n\t"
                 "rdtsc\n\t"
                 : "=a"(lo), "=d"(hi)
                 :: "memory");
    return ((uint64_t)hi << 32) | lo;
}
```

#### Dispatch Timing Loop:
```c
uint64_t t_start = rdtsc_ordered();
struct task_struct *next = neuroos_select_next_task(rq);
uint64_t t_eval = rdtsc_ordered() - t_start;
/* Target: t_eval < 150 cycles (~45-50 ns @ 3.0 GHz) */
```

---

### 3. Workload Synthesis & Datasets

#### 3.1 Synthetic Heavy-Tailed Workloads
- **Burst Duration Distribution**: Generated via Pareto distribution:
  $$P(X > x) = \left( \frac{x_m}{x} \right)^\alpha, \quad \alpha \in [1.1, 1.8]$$
  to replicate heavy-tailed computational characteristics where 20% of tasks consume 80% of CPU cycles.
- **Arrival Process**: Poisson arrival process with offered load factors:
  $$\rho = \frac{\lambda \cdot \mathbb{E}[b]}{\mu} \in [0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.98]$$

#### 3.2 Real-World Trace Replay
- **Google Borg Cluster Trace Slice**:
  - Filtered uniprocessor task slice extracting task arrival timestamps, CPU execution runtimes, and memory utilization profiles.
- **SPEC CPU2017 & MiBench Benchmarks**:
  - Multi-phase execution profiles switching dynamically between compute-intensive (e.g., `gcc`, `lbm`) and memory/cache-intensive phases (e.g., `mcf`, `milc`).
  - Traced via hardware PMU counters (`perf stat`) to supply realistic telemetry streams.

#### 3.3 Adversarial Edge-Case Workloads
- **Pathological Convoy Triggers**: A continuous stream of short interactive jobs ($b_{short} = 10\ \mu\text{s}$) queued directly behind an immense compute job ($b_{long} = 10^5\ \mu\text{s}$). Tests anti-convoy preemption.
- **Alternating Fragmentation Triggers**: Rapid interleaved allocation and deallocation of alternating odd-even small blocks ($4\text{ KB}$ and $64\text{ KB}$) to induce maximal address slivers in naive allocators. Tests lifetime-affinity clustering.

---

### 4. Baseline Algorithms for Comparison

| Domain | Baseline Strategy | Category | Theoretical Complexity | Known Vulnerability |
|---|---|---|---|---|
| CPU Scheduling | **FCFS** | Non-preemptive | $O(1)$ | Convoy effect; high waiting times for short jobs |
| CPU Scheduling | **SJF** | Non-preemptive | $O(\log n)$ | Starvation of long jobs; requires perfect burst oracle |
| CPU Scheduling | **SRTF** | Preemptive | $O(\log n)$ | High context-switch frequency; estimation errors |
| CPU Scheduling | **Round Robin (5ms, 20ms)** | Preemptive | $O(1)$ | Cache thrashing at low $q$; FCFS convergence at high $q$ |
| CPU Scheduling | **MLFQ** | Preemptive | $O(1)$ amortized | Gaming vulnerability; priority tuning complexity |
| CPU Scheduling | **NeuroOS-Lite** | Preemptive | $O(1)$ SIMD | Bounded by guardrail fallback ($< 50\text{ ns}$) |
| Memory Allocation | **Fixed Partitioning (MFT)** | Static Table | $O(1)$ | Severe internal fragmentation |
| Memory Allocation | **Dynamic Best-Fit** | Variable (MVT) | $O(n)$ or $O(\log n)$ | Severe external fragmentation slivers |
| Memory Allocation | **Dynamic First-Fit** | Variable (MVT) | $O(n)$ | Front-end memory accumulation; slivers |
| Memory Allocation | **Binary Buddy System** | Power-of-Two | $O(\log M)$ | Internal fragmentation up to 49.9% for $2^k + 1$ |
| Memory Allocation | **NeuroOS-Lite** | Lifetime-Affinity | $O(1) / O(\log N)$ | Bounded by Buddy fallback |

---

### 5. Ablation Studies

1. **Ablation 1: Impact of Hardware PMU Counters**:
   - Compare full model (burst history + PMU cache/branch deltas) vs. reduced model (raw burst history alone).
   - Quantifies the value of hardware performance counters in predicting phase transitions.
2. **Ablation 2: Quantization Precision Penalty**:
   - Compare unquantized FP32 reference network against 8-bit quantized integer SIMD kernel.
   - Measure degradation in decision quality vs. 100x speedup in evaluation latency.
3. **Ablation 3: Power & Energy Overhead**:
   - Monitor total system power consumption using Intel RAPL counters (CPU) and NVIDIA NVML (GPU).
   - Evaluate whether GPU training energy is amortized over millions of fast-path dispatch decisions.

---

### 6. Publication Visualization Specifications (IEEE Transactions Layout)

Four primary publication figures will be produced from empirical benchmark outputs:
1. **CDFs of Normalized Turnaround Time ($NTAT$) & Waiting Time**:
   - X-axis: $NTAT$ (log scale from $1.0$ to $100.0$).
   - Y-axis: Cumulative probability ($0.0$ to $1.0$).
   - Curves: FCFS, SJF, SRTF, RR-5ms, RR-20ms, MLFQ, NeuroOS-Lite.
2. **Pareto Trade-Off Frontier (Decision Overhead vs. Mean Waiting Time)**:
   - X-axis: Mean subsystem decision latency (ns, log scale $10^1$ to $10^5\text{ ns}$).
   - Y-axis: Average waiting time (ms).
3. **Dynamic Memory Allocation Heatmap (Space vs. Time)**:
   - X-axis: Timeline ($10^6$ allocation events).
   - Y-axis: Physical heap address space ($0$ to $M_{total}$).
   - Visualizing Best-Fit fragmentation slivers vs. NeuroOS-Lite lifetime clustering bands.
4. **Overhead & Energy Stacked Bar Chart**:
   - Latency breakdown: Context switch saving/restoring, in-kernel evaluation ($< 45\text{ ns}$), ring buffer sync, GPU training amortization.
