# Lock-Free SPSC Circular Ring Buffer

This directory contains the single-producer single-consumer circular buffer shared between kernel space and the user-space GPU daemon.

## Status: `SCAFFOLDED` (Phase 1, Week 1)

## Design Invariants
- Zero memory allocations during runtime.
- Wait-free producer (kernel context-switch hook).
- Cacheline-padded head and tail pointers (64-byte alignment) to eliminate false sharing.
- Power-of-two buffer capacity for bitmask index wrapping.
- See `include/ring_buffer.h` for C interfaces.
