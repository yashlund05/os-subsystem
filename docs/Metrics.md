# Metrics & Evaluation Formulas

This document provides formal mathematical definitions and operational formulations for all evaluation metrics tracked in NeuroOS-Lite.

---

### 1. CPU Scheduling Metrics

```text
                     +---------------------------------------+
                     |            Turnaround Time            |
                     | <-----------------------------------> |
Arrival              | Execution   Wait   Execution          | Completion
  t_a                | (Burst b_1) (w_1)  (Burst b_2)        |     t_c
───┼─────────────────+===========+───────+===========+───────+───────> Time
   |                 |                               |
   | <- Response ->  |                               |
   |   Latency (R)   |                               |
```

#### 1.1 Turnaround Time ($TAT$) & Normalized Turnaround Time ($NTAT$)
The total elapsed time from task arrival $t_a(i)$ to task completion $t_c(i)$:
$$TAT_i = t_c(i) - t_a(i)$$

To prevent long compute bursts from skewing arithmetic averages and to fairly evaluate short interactive jobs, we compute Normalized Turnaround Time ($NTAT$):
$$NTAT_i = \frac{TAT_i}{\sum_{k} b_i(k)}$$
where $\sum_k b_i(k)$ is the total CPU burst requirement of task $i$. An ideal scheduler achieves $NTAT \to 1.0$.

#### 1.2 Waiting Time ($WT$)
The cumulative duration spent in the ready queue $\mathcal{Q}_R$ awaiting CPU dispatch:
$$WT_i = TAT_i - \sum_{k} b_i(k) = t_c(i) - t_a(i) - \sum_{k} b_i(k)$$

#### 1.3 Response Time ($RT$)
The duration from initial task arrival to first dispatch tick:
$$RT_i = t_{\text{first\_dispatch}}(i) - t_a(i)$$

#### 1.4 Tail Latency Metrics ($P_{95}, P_{99}, P_{99.9}$)
For a population of $N$ tasks sorted by waiting time $WT_{(1)} \le WT_{(2)} \le \dots \le WT_{(N)}$:
- $P_{95} = WT_{(\lceil 0.95 \cdot N \rceil)}$
- $P_{99} = WT_{(\lceil 0.99 \cdot N \rceil)}$
- $P_{99.9} = WT_{(\lceil 0.999 \cdot N \rceil)}$

These tail percentiles prove anti-starvation guarantees under bursty and heavy-tailed workloads.

#### 1.5 Context-Switch Frequency & Overhead Ratio ($\Phi_{overhead}$)
A critical trade-off in preemptive scheduling is context-switch overhead $\bar{t}_{ctx\_save}$ (register save/restore, TLB flushes, cache invalidation) versus latency responsiveness:
$$\Phi_{overhead} = \frac{N_{ctx\_switches} \cdot \bar{t}_{ctx\_save} + \sum_{j} t_{inference}(j)}{T_{total\_makespan}}$$
where:
- $N_{ctx\_switches}$: Total number of preemptive context switches.
- $\bar{t}_{ctx\_save}$: Average cost of a context switch (~0.8 to 2.5 µs on x86_64).
- $t_{inference}$: Execution time of the in-kernel micro-inference core ($\le 45\text{ ns}$).
- $T_{total\_makespan}$: Total workload execution duration.

This metric guarantees that the AI scheduler does not win on queue theory while losing to cycle-level CPU overheads.

---

### 2. Memory Subsystem Metrics

#### 2.1 External Fragmentation ($\text{Frag}_{external}$)
Quantifies the fraction of unallocated memory that cannot be satisfied due to address space scattering:
$$\text{Frag}_{external}(t) = 1 - \frac{\max_{B \in \mathcal{B}_{free}(t)} \text{Size}(B)}{\sum_{B \in \mathcal{B}_{free}(t)} \text{Size}(B)}$$
where:
- $\mathcal{B}_{free}(t)$: Set of all currently unallocated free memory blocks at time $t$.
- $\max_{B \in \mathcal{B}_{free}(t)} \text{Size}(B)$: Size of the single largest contiguous free block.

When all free memory exists in one contiguous block, $\text{Frag}_{external} = 0$. When free memory is fractured into small slivers, $\text{Frag}_{external} \to 1.0$.

#### 2.2 Internal Fragmentation ($\text{Frag}_{internal}$)
Quantifies wasted memory inside allocated blocks due to alignment or power-of-two rounding (e.g., in Buddy or Fixed Partitioning allocators):
$$\text{Frag}_{internal}(t) = \frac{\sum_{k \in \mathcal{A}(t)} (\text{Allocated}(k) - \text{Requested}(k))}{\sum_{k \in \mathcal{A}(t)} \text{Allocated}(k)}$$
where $\mathcal{A}(t)$ is the set of currently active allocations at time $t$.

#### 2.3 Buffer Utilization ($\eta_{buf}$)
The ratio of active application payload bytes to total heap footprint:
$$\eta_{buf}(t) = \frac{\sum_{k \in \mathcal{A}(t)} \text{Requested}(k)}{M_{total}}$$

---

### 3. Subsystem Decision & Inference Latency
Measured in physical clock cycles using ordered Time Stamp Counter (`rdtsc_ordered`) on an isolated host core:
$$\text{Latency}_{inference} = \frac{\text{TSC}_{end} - \text{TSC}_{start}}{f_{CPU}}$$
- Target latency: $\le 45\text{ ns}$ (< 150 cycles @ 3.0 GHz).
- Hard guardrail limit: $< 50\text{ ns}$.
