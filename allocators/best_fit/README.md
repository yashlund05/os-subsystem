# Dynamic Best-Fit Memory Allocator Baseline

## Theoretical Profile
- **Category**: Variable Partitioning (MVT)
- **Complexity**: $O(n)$ scan or $O(\log n)$ balanced tree
- **Pathology / Vulnerability**: Extreme external fragmentation; leaves behind tiny, unusable memory holes across the address space.
- **Status**: `IMPLEMENTED` (Completed Phase 1, Week 2)
