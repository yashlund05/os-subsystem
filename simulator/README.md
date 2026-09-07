# Cycle-Accurate Simulator Subsystem

This directory contains the cycle-accurate uniprocessor discrete-event simulation engine for CPU scheduling and dynamic memory partitioning.

## Status: `SCAFFOLDED` (Phase 1, Week 1–2)

## Modules
- `simulator/scheduling/`: Discrete-event uniprocessor CPU scheduler queue, context-switch accounting, and metrics calculation.
- `simulator/memory/`: Dynamic heap allocation model, external and internal fragmentation trackers, and allocation lifetime monitors.
- `simulator/workloads/`: Workload generators including synthetic Pareto bursts, Poisson arrivals, and cluster trace replay.
