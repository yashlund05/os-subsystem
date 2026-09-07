# Shortest Remaining Time First (SRTF) Scheduler Baseline

## Theoretical Profile
- **Category**: Preemptive
- **Complexity**: $O(\log n)$ balanced tree / min-heap
- **Pathology / Vulnerability**: High context-switch frequency; estimation error cascades; starvation of long jobs under high load.
- **Status**: `SCAFFOLDED` (Scheduled for Phase 1, Week 2 implementation)
