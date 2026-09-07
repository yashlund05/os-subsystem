# User-Space Telemetry Ingestion Daemon

This directory contains the user-space daemon that consumes telemetry events from the lock-free SPSC ring buffer.

## Status: `NOT IMPLEMENTED` (Scheduled for Phase 1, Week 1)

## Intended Responsibilities
- Poll or wait on lock-free SPSC ring buffer without introducing lock contention with kernel producers.
- Parse 16-byte `task_telemetry` structs into training batches.
- Compute rolling statistics (IPC, mean cache miss deltas, burst variances) for feature pipelines.
