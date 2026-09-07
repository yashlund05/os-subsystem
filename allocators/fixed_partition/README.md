# Fixed Partitioning (MFT) Memory Allocator Baseline

## Theoretical Profile
- **Category**: Static Partition Table
- **Complexity**: $O(1)$ bitmap or slot table lookup
- **Pathology / Vulnerability**: Severe internal fragmentation when requested sizes do not align with static slot boundaries.
- **Status**: `SCAFFOLDED` (Scheduled for Phase 1, Week 2 implementation)
