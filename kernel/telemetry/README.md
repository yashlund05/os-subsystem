# Kernel Telemetry Producer

This directory contains kernel-space telemetry instrumentation hooks.

## Status: `NOT IMPLEMENTED` (Scheduled for Phase 1, Week 1)

## Intended Responsibilities
- Intercept context-switch events and query hardware PMU counters.
- Populate the 16-byte `struct task_telemetry` (`kernel/include/telemetry_event.h`).
- Push events to the lock-free SPSC circular ring buffer with wait-free memory barriers.
- Record telemetry drop counts if buffer is full without blocking kernel execution.
