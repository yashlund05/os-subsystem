# Kernel Core Header Interfaces (Phase 1 & Phase 2)

Core C header declarations and structs for the NeuroOS-Lite in-kernel subsystem.

- `telemetry_event.h`: 16-byte packed telemetry event struct conforming to Linux perf/eBPF formats.
- `neuroos_kernel.h`: 16->8->1 integer MLP forward-pass signature and constant declarations.
- `neuroos_weights.h`: Quantized INT8 weight matrices and bias vectors for the student policy.
- `neuroos_lut.h`: Fast lookup table (LUT) tables for activation functions without floating point.

**Status**: `IMPLEMENTED` (Completed Phase 1, Week 1 & Phase 2, Week 4)
