# User-Space Policy Management & Shared Tables

This directory manages policy serialization, version tracking, and atomic shared memory synchronization.

## Status: `SCAFFOLDED` (Scheduled for Phase 2, Week 4)

## Intended Responsibilities
- Serialize distilled model parameters into `struct neuroos_quantized_policy`.
- Maintain double-buffered shared memory segment mapped between user space and kernel space.
- Perform low-frequency (~1–5 Hz) atomic pointer swaps without stalling or interrupting active dispatch decisions.
