"""Phase 2-4 regression tests (no torch/gym required)."""

import numpy as np


def test_numpy_student_forward_shape():
    from ml.distillation.distiller import numpy_student_forward

    rng = np.random.default_rng(0)
    params = {
        "layer_0_weight": rng.standard_normal((8, 16)),
        "layer_0_bias": np.zeros(8),
        "layer_1_weight": rng.standard_normal((1, 8)),
        "layer_1_bias": np.zeros(1),
    }
    feats = rng.random((5, 16)) * 5.0
    out = numpy_student_forward(params, feats)
    assert out.shape == (5,)


def test_quantize_roundtrip_and_checksum():
    from ml.quantization.quantize import c_header_bytes, quantized_forward_int, quantize_student

    rng = np.random.default_rng(1)
    params = {
        "layer_0_weight": rng.standard_normal((8, 16)) * 0.3,
        "layer_0_bias": rng.standard_normal(8) * 0.1,
        "layer_1_weight": rng.standard_normal((1, 8)) * 0.3,
        "layer_1_bias": rng.standard_normal(1) * 0.1,
    }
    qpolicy, meta = quantize_student(params, version=3)
    assert qpolicy["w1"].dtype == np.int8
    assert qpolicy["b1"].dtype == np.int16
    assert qpolicy["w2"].dtype == np.int8
    assert qpolicy["b2"].dtype == np.int32
    assert meta["version"] == 3
    assert meta["checksum"] != 0
    feats = rng.random((8, 16)) * 5.0
    out = quantized_forward_int(qpolicy, meta, feats)
    assert out.shape == (8,)
    assert np.all(np.isfinite(out))
    blob = c_header_bytes(qpolicy, meta)
    assert len(blob) == 128 + 16 + 8 + 4 + 4


def test_quantum_lut_monotonic():
    from ml.quantization.lut import build_quantum_lut, lifetime_to_band

    lut = build_quantum_lut(q_min_us=1000, q_max_us=50000, num_bins=8)
    assert lut.lookup(0) == 1000
    assert lut.lookup(7) == 50000
    assert lut.score_to_quantum(-10.0) <= lut.score_to_quantum(10.0)
    assert lifetime_to_band(1000) < lifetime_to_band(1000000)


def test_policy_table_publish_read():
    import numpy as np

    from ml.quantization.quantize import quantize_student
    from userspace.policy.policy_table import DoubleBufferedPolicyTable

    rng = np.random.default_rng(2)
    params = {
        "layer_0_weight": rng.standard_normal((8, 16)) * 0.2,
        "layer_0_bias": np.zeros(8),
        "layer_1_weight": rng.standard_normal((1, 8)) * 0.2,
        "layer_1_bias": np.zeros(1),
    }
    qpolicy, meta = quantize_student(params, version=1)
    tbl = DoubleBufferedPolicyTable()
    assert tbl.version == 0
    v = tbl.publish(qpolicy, meta)
    assert v == 1
    snap = tbl.read()
    assert snap is not None and snap.checksum == meta["checksum"]


def test_neuroos_scheduler_no_oracle():
    from benchmarks.workloads.suites import make_scheduling_workload
    from schedulers.neuroos_lite.scheduler import NeuroOSLiteScheduler
    from simulator.scheduling.engine import SchedulingSimulationEngine

    rng = np.random.default_rng(3)
    params = {
        "layer_0_weight": rng.standard_normal((8, 16)) * 0.3,
        "layer_0_bias": np.zeros(8),
        "layer_1_weight": rng.standard_normal((1, 8)) * 0.3,
        "layer_1_bias": np.zeros(1),
    }
    sched = NeuroOSLiteScheduler(student_params=params)
    wl = make_scheduling_workload("pareto_bursts", 0.5, num_tasks=20, seed=3)
    eng = SchedulingSimulationEngine(scheduler=sched, context_switch_overhead_us=2)
    done, metrics = eng.run(wl)
    assert len(done) == 20
    assert metrics.mean_waiting_time_us >= 0


def test_lifetime_allocator_metrics_range():
    from allocators.lifetime_affinity.allocator import LifetimeAffinityAllocator
    from benchmarks.workloads.suites import make_memory_trace
    from benchmarks.runner import BenchmarkRunner

    alloc = LifetimeAffinityAllocator(total_heap_bytes=4 * 1024 * 1024)
    events = make_memory_trace("alternating_odd_even_trigger", num_pairs=10)
    res = BenchmarkRunner.run_memory_benchmark(
        allocator=alloc, events=events, experiment_id="t", workload_type="t"
    )
    assert 0.0 <= res.metrics["external_fragmentation"] <= 1.0
    assert 0.0 <= res.metrics["internal_fragmentation"] <= 1.0


def test_scheduling_and_memory_sweeps_small():
    from benchmarks.memory.sweep import run_memory_matrix
    from benchmarks.scheduling.sweep import run_scheduling_matrix

    s = run_scheduling_matrix(
        num_tasks=10, seed=1, profiles=["pareto_bursts"], load_factors=[0.5]
    )
    assert len(s) == 7  # 7 algorithms
    m = run_memory_matrix(
        heap_bytes=4 * 1024 * 1024, traces=["alternating_odd_even_trigger"], num_pairs=5
    )
    assert len(m) == 5  # 5 allocators


def test_ablations_smoke():
    from benchmarks.ablations.ablations import ablation_power, ablation_quantization

    q = ablation_quantization(seed=1)
    assert q["pearson_r"] > 0.9
    p = ablation_power()
    assert "rapl" in p and "nvml" in p
