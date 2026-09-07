# In-Kernel Micro-Inference Core

This directory contains the branchless, quantized integer inference engine evaluated directly in the OS dispatch path.

## Status: `NOT IMPLEMENTED` (Scheduled for Phase 2, Week 4)

## Intended Responsibilities
- 2-layer quantized MLP ($16 \to 8 \to 1$).
- Pure signed integer arithmetic (`int8_t` weights, `int16_t` hidden activations/biases, `int32_t` accumulators).
- Zero floating-point operations (FPU register saving disabled).
- SIMD acceleration: AVX2 (`_mm256_maddubs_epi16`, `_mm256_add_epi16`) on x86_64, NEON (`vdot_s8`) on ARM64.
- Latency target: $\le 45\text{ ns}$ (< 150 clock cycles @ 3.0 GHz).
