# Shortest Job First (SJF) Scheduler Baseline

## Theoretical Profile
- **Category**: Non-Preemptive
- **Complexity**: $O(\log n)$ priority queue insertion
- **Pathology / Vulnerability**: Starvation of long bursts; impossible perfect oracle assumption for future burst estimation in real OS.
- **Status**: `IMPLEMENTED` (Completed Phase 1, Week 2)
