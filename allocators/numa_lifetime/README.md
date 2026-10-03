# NUMA-Aware Lifetime Affinity Memory Allocator (Phase 6)

Extends the single-domain lifetime-affinity clustering allocator to multi-socket NUMA systems.

## Key Mechanisms
- **Per-Node Lifetime Bands**: Partitions memory into independent lifetime classes on each NUMA node (`<5ms`, `<20ms`, `<100ms`, `<500ms`, `>=500ms`).
- **Thread Locality Priority**: Resolves requests from the calling thread's local NUMA node first, avoiding memory bus interconnect hops.
- **Hierarchical Neighbor Probing**: If a lifetime band is full, explores neighbor lifetime bands on the *same* NUMA node before querying remote sockets.
- **Buddy Fallback**: Preserves fail-safe allocation guarantees when all local and remote bands are saturated.

**Status**: `IMPLEMENTED` (Completed Phase 6)
