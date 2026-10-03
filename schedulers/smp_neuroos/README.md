# SMP NeuroOS-Lite Multi-Core Scheduler (Phase 6)

Multi-core extension of the NeuroOS-Lite learned scheduler with NUMA awareness and work-stealing.

## Features
- **Cache-Warmth CPU Placement**: Preserves cache warmth for recently executing tasks, dampening intra-core migration latency.
- **NUMA-Aware Distance Penalties**: Accounts for inter-socket interconnect distances (QPI/UPI) during initial enqueue and work stealing.
- **Hierarchical Work-Stealing**: Idle cores query intra-NUMA cores before probing remote NUMA sockets, preventing ping-pong migration overhead.
- **Online Drift Self-Correction**: Integrates continuous integer residual error feedback, suppressing unnecessary MLFQ fallback trips.

**Status**: `IMPLEMENTED` (Completed Phase 6)
