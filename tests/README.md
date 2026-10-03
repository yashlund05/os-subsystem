# NeuroOS-Lite Test Suite

Complete test hierarchy for NeuroOS-Lite covering unit, integration, performance regression, and kernel C header verification.

## Directory Structure

- `unit/`: Unit tests for schedulers, allocators, simulation engine, ML models, quantization, and metrics.
- `integration/`: End-to-end integration tests verifying ring-buffer, guardrail trip/recovery, and scheduler pipeline dispatch.
- `performance/`: Microsecond-scale latency regression tests verifying in-kernel quantized forward bounds (<100 µs) and student model throughput.
- `kernel/`: C harness tests for packed struct layouts, SIMD topology constants, and algorithm enums.

**Status**: `IMPLEMENTED` (Completed Phases 1–5)
