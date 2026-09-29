# Lifetime-Affinity Memory Allocator (Phase 3, Week 6)

Colocates allocations with correlated `tau_k` into contiguous bands.
Bands by `tau_k`: `<5ms`, `<20ms`, `<100ms`, `<500ms`, `>=500ms`.
Fallback: Buddy allocator if band + neighbor probe fails.
