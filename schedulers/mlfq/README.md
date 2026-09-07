# Multi-Level Feedback Queue (MLFQ) Scheduler Baseline

## Theoretical Profile
- **Category**: Preemptive
- **Complexity**: $O(1)$ amortized multi-queue head pop
- **Pathology / Vulnerability**: Gaming vulnerabilities (yielding right before quantum expiration); tuning complexity of priority levels, quantum scalers, and starvation boost intervals.
- **Status**: `SCAFFOLDED` (Scheduled for Phase 1, Week 2 implementation)
