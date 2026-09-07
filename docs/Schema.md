# Data Schemas & Interface Specifications

This document defines the authoritative memory structures, wire formats, and API schemas used across NeuroOS-Lite.

---

### 1. Telemetry Event Schema (`task_telemetry`)

The kernel streams telemetry over the lock-free SPSC ring buffer using a packed 16-byte struct:

```c
struct __attribute__((packed)) task_telemetry {
    uint32_t pid;                  /* Task identifier (32 bits) */
    uint16_t elapsed_us;            /* Elapsed execution time in microsecond ticks (16 bits) */
    uint16_t cache_misses_delta;   /* L1/LLC cache miss delta since last switch (16 bits) */
    uint16_t branch_mispred_delta; /* Branch misprediction delta (16 bits) */
    uint16_t mem_footprint_kb;     /* Task memory working set in KB (16 bits) */
    uint16_t flags;                /* State flags: burst phase, priority, guardrail status (16 bits) */
};
```

#### Field Details:
| Field | Type | Bits | Units / Semantics |
|---|---|---|---|
| `pid` | `uint32_t` | 32 | Process / Task ID |
| `elapsed_us` | `uint16_t` | 16 | Execution duration in $\mu\text{s}$ during current quantum |
| `cache_misses_delta` | `uint16_t` | 16 | Hardware PMU L1/LLC data cache misses since last switch |
| `branch_mispred_delta` | `uint16_t` | 16 | Hardware PMU branch mispredictions since last switch |
| `mem_footprint_kb` | `uint16_t` | 16 | Current RSS memory footprint in KB |
| `flags` | `uint16_t` | 16 | State bitmask (see flag definitions below) |

#### Telemetry Flags:
- `0x0001` (`TELEMETRY_FLAG_BURST_START`): Task newly dispatched after idle / wait.
- `0x0002` (`TELEMETRY_FLAG_BURST_END`): Current CPU burst completed.
- `0x0004` (`TELEMETRY_FLAG_IO_BLOCKED`): Task blocked on I/O or voluntary wait.
- `0x0008` (`TELEMETRY_FLAG_PREEMPTED`): Task preempted by quantum expiration or higher-priority arrival.
- `0x0010` (`TELEMETRY_FLAG_GUARDRAIL_TRIP`): Decision bypassed due to guardrail trigger.
- `0x0020` (`TELEMETRY_FLAG_FALLBACK_ACTIVE`): Classical fallback scheduler active.

---

### 2. Task Descriptor State Schema

```c
struct task_descriptor {
    uint32_t pid;                   /* Unique process identifier */
    uint64_t arrival_time_us;       /* Absolute simulation/system arrival timestamp */
    uint64_t total_burst_us;         /* Total duration of CPU execution requested */
    uint64_t executed_burst_us;      /* Execution time accumulated so far */
    uint64_t remaining_burst_us;     /* Residual CPU burst duration */
    uint64_t last_dispatch_time_us;  /* Timestamp of most recent dispatch */
    uint64_t waiting_time_us;        /* Accumulated wait time in ready queue */
    uint32_t context_switches;      /* Total context switches experienced */
    uint32_t priority_level;         /* Static or base priority level */
    void *priv;                     /* Subsystem private metadata */
};
```

---

### 3. Quantized Neural Policy Schema (16 -> 8 -> 1 MLP)

```c
#define NEUROOS_INPUT_DIM   16
#define NEUROOS_HIDDEN_DIM  8
#define NEUROOS_OUTPUT_DIM  1

struct neuroos_quantized_policy {
    int8_t  w1[NEUROOS_INPUT_DIM * NEUROOS_HIDDEN_DIM];  /* 16x8 = 128 bytes (int8) */
    int16_t b1[NEUROOS_HIDDEN_DIM];                     /* 8 x 2 = 16 bytes (int16) */
    int8_t  w2[NEUROOS_HIDDEN_DIM * NEUROOS_OUTPUT_DIM]; /* 8x1 = 8 bytes (int8) */
    int32_t b2[NEUROOS_OUTPUT_DIM];                     /* 1 x 4 = 4 bytes (int32) */
    uint32_t version;                                   /* Monotonic version counter */
    uint32_t checksum;                                  /* CRC32 integrity checksum */
};
```
Total policy footprint: $< 200\text{ bytes}$, easily resident in L1 instruction/data cache.

---

### 4. Scheduling Decision Schema

```c
struct neuroos_decision {
    uint32_t selected_pid;     /* Target PID picked for execution */
    uint32_t quantum_us;        /* Allocated time slice in microseconds [q_min, q_max] */
    bool fallback_engaged;     /* True if decision was made by deterministic fallback */
    uint16_t reason_code;      /* 0: Normal neural inference, 1: Queue saturation, 2: Drift trip */
};
```

---

### 5. Memory Allocation Request & Block Schemas

```c
struct alloc_request {
    uint32_t request_id;              /* Unique request ID */
    uint32_t task_pid;                /* Requesting task PID */
    size_t size_bytes;                /* Requested memory size */
    uint32_t predicted_lifetime_us;   /* Anticipated lifetime tau_k for affinity binning */
};

struct alloc_handle {
    uintptr_t base_address;           /* Starting memory address */
    size_t allocated_size_bytes;      /* Actual block size allocated */
    size_t requested_size_bytes;      /* Requested block size */
    uint32_t block_id;                /* Internal heap block index */
    bool success;                     /* Allocation success flag */
};
```

---

### 6. Benchmark Result Metric Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NeuroOSBenchmarkResult",
  "type": "object",
  "properties": {
    "experiment_id": { "type": "string" },
    "subsystem": { "type": "string", "enum": ["scheduling", "memory", "overhead"] },
    "algorithm": { "type": "string" },
    "workload_type": { "type": "string" },
    "load_factor": { "type": "number" },
    "metrics": {
      "type": "object",
      "properties": {
        "mean_turnaround_time_us": { "type": "number" },
        "mean_normalized_turnaround_time": { "type": "number" },
        "mean_waiting_time_us": { "type": "number" },
        "p95_waiting_time_us": { "type": "number" },
        "p99_waiting_time_us": { "type": "number" },
        "p999_waiting_time_us": { "type": "number" },
        "total_context_switches": { "type": "integer" },
        "external_fragmentation": { "type": "number" },
        "internal_fragmentation": { "type": "number" },
        "buffer_utilization": { "type": "number" },
        "mean_inference_latency_ns": { "type": "number" },
        "guardrail_fallback_trips": { "type": "integer" }
      },
      "required": ["mean_turnaround_time_us", "mean_waiting_time_us", "total_context_switches"]
    }
  },
  "required": ["experiment_id", "subsystem", "algorithm", "workload_type", "load_factor", "metrics"]
}
```
