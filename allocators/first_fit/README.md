# Dynamic First-Fit Memory Allocator Baseline

## Theoretical Profile
- **Category**: Variable Partitioning (MVT)
- **Complexity**: $O(n)$ linear free-list scan
- **Pathology / Vulnerability**: Front-end memory accumulation; small unusable slivers accumulate near the beginning of the free list.
- **Status**: `SCAFFOLDED` (Scheduled for Phase 1, Week 2 implementation)
