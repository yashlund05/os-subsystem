# Kernel `sched_ext` Subsystem

This directory contains the Linux `sched_ext` (extensible scheduler core via BPF) driver and dispatch hooks for NeuroOS-Lite.

## Status: `NOT IMPLEMENTED` (Scheduled for Phase 3, Week 5)

## Intended Responsibilities
- Implement BPF dispatch hooks: `ops.select_cpu`, `ops.enqueue`, `ops.dispatch`.
- Intercept context-switch events and invoke the in-kernel micro-inference engine.
- Apply dynamic quantum sizes $\Delta t_q \in [q_{min}, q_{max}]$ returned by the policy.
- Bounded execution time: $< 50\text{ ns}$ per dispatch decision.
- Immediate fail-safe routing to $O(1)$ Red-Black tree / MLFQ if guardrails trip.
