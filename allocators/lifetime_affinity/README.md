# Lifetime-Affinity Memory Allocator (Phase 3, Week 6)

## Theoretical Profile
- **Category**: Lifetime-Correlated Multi-Band Partitioning with Buddy Fallback
- **Complexity**: $O(1)$ fast-path band lookup, $O(\log M)$ buddy fallback
- **Design**: Colocates allocations with correlated predicted lifetimes `tau_k` into contiguous bands (`<5ms`, `<20ms`, `<100ms`, `<500ms`, `>=500ms`) to minimize external fragmentation.
- **Status**: `IMPLEMENTED` (Completed Phase 3, Week 6)

