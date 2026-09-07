# First-Come, First-Served (FCFS) Scheduler Baseline

## Theoretical Profile
- **Category**: Non-Preemptive
- **Complexity**: $O(1)$ queue head dispatch
- **Pathology / Vulnerability**: Convoy effect; short interactive tasks suffer severe waiting times when stuck behind long compute bursts.
- **Status**: `SCAFFOLDED` (Scheduled for Phase 1, Week 2 implementation)
