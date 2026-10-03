# Unit Test Suite (Phases 1–4)

Comprehensive unit test coverage for individual components across simulator, ML training, quantization, allocators, schedulers, and metrics.

- `test_allocators.py`: Memory allocation unit tests (Buddy, First-Fit, Best-Fit, Fixed Partition).
- `test_schedulers.py`: Classic scheduler algorithm tests (FCFS, SJF, SRTF, RR, MLFQ).
- `test_simulation_engine.py`: Discrete event engine and tick execution.
- `test_simulator.py`: High-level simulation harness verification.
- `test_workloads.py`: Synthetic Pareto/Poisson and Borg trace workload generators.
- `test_metrics.py`: Metric collection, throughput, fragmentation, and latency calculations.
- `test_feature_histogram.py`: Observation feature extraction and histogram verification.
- `test_drl_training.py`: PPO policy training pipeline and environment step transitions.
- `test_scheduler_env.py`: Gymnasium scheduler environment actions, observations, and rewards.
- `test_phase24.py`: Phase 2 distillation and Phase 4 model verification.
- `test_quantization.py`: INT8 weight quantization, LUT generation, and calibration.

**Status**: `IMPLEMENTED` (Completed Phases 1–4)
