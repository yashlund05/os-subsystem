"""Unit tests for Phase 4 Quantization Pipeline and Guardrail Engine."""

from pathlib import Path

import numpy as np

from quantization.guardrail import (
    FallbackReason,
    GuardrailEngine,
    c_guardrail_check,
)
from quantization.int8_forward import (
    int8_forward_batch,
    int8_forward_single,
    requantize_acc1_to_int8,
)
from quantization.lut_generator import QuantumLUT, generate_c_lut_header
from quantization.quantize import (
    calibrate_input_scale,
    export_c_weights_header,
    quantize_student_policy,
)


def test_quantize_pipeline_and_checksum(tmp_path: Path):
    rng = np.random.default_rng(42)
    calib = rng.random((500, 16)) * 2.0
    s_x = calibrate_input_scale(calib, percentile=98.0)
    assert s_x.shape == (16,)
    assert np.all(s_x > 0.0)

    float_weights = {
        "w1": rng.standard_normal((8, 16)) * 0.3,
        "b1": rng.standard_normal(8) * 0.1,
        "w2": rng.standard_normal((1, 8)) * 0.3,
        "b2": np.array([0.05]),
    }

    qpolicy, meta = quantize_student_policy(float_weights, s_x=s_x, version=1)
    assert qpolicy["w1"].dtype == np.int8
    assert qpolicy["b1"].dtype == np.int16
    assert qpolicy["w2"].dtype == np.int8
    assert qpolicy["b2"].dtype == np.int32
    assert meta["checksum"] > 0

    c_header = tmp_path / "neuroos_weights.h"
    export_c_weights_header(qpolicy, meta, output_path=str(c_header))
    assert c_header.exists()
    content = c_header.read_text(encoding="utf-8")
    assert "neuroos_student_policy" in content
    assert "neuroos_scale_s_x" in content


def test_int8_forward_batch_matches_single():
    rng = np.random.default_rng(123)
    w1 = rng.integers(-128, 127, size=(8, 16), dtype=np.int8)
    b1 = rng.integers(-1000, 1000, size=(8,), dtype=np.int16)
    w2 = rng.integers(-128, 127, size=(1, 8), dtype=np.int8)
    b2 = np.array([500], dtype=np.int32)

    features = rng.integers(-128, 127, size=(20, 16), dtype=np.int8)
    batch_out = int8_forward_batch(features, w1, b1, w2, b2)
    assert batch_out.shape == (20,)

    for i in range(20):
        single_out = int8_forward_single(features[i], w1, b1, w2, b2)
        assert single_out == batch_out[i], (
            f"Mismatch at idx {i}: single={single_out}, batch={batch_out[i]}"
        )


def test_requantize_acc1_bit_identical():
    # Exact C integer logic simulation:
    # int64_t prod = (int64_t)acc1 * 3333 + 524288;
    # int32_t scaled = (int32_t)(prod >> 20);
    # if (scaled > 127) scaled = 127;
    def c_requant(acc1_val: int) -> int:
        if acc1_val <= 0:
            return 0
        prod = acc1_val * 3333 + (1 << 19)
        scaled = prod >> 20
        return min(127, max(0, scaled))

    # Test edge cases: negative, zero, boundary at 39953, and large overflows
    test_values = list(range(-500, 50000, 17)) + [0, 1, 314, 39953, 40000, 65535, 100000]
    for val in test_values:
        py_res = int(requantize_acc1_to_int8(val))
        c_res = c_requant(val)
        assert py_res == c_res, f"Bit mismatch at {val}: py={py_res}, c={c_res}"

    # Batch test
    arr = np.array(test_values, dtype=np.int32)
    py_batch = requantize_acc1_to_int8(arr)
    for idx, val in enumerate(test_values):
        assert py_batch[idx] == c_requant(val)


def test_quantum_lut_monotonic_and_bounds(tmp_path: Path):
    lut = QuantumLUT(
        q_min_us=1000, q_max_us=50000, num_entries=256, score_min=-1000, score_max=1000
    )
    assert lut.lookup(0) == 1000
    assert lut.lookup(255) == 50000

    # Monotonicity
    assert lut.score_to_quantum(-1500) == 1000
    assert lut.score_to_quantum(1500) == 50000
    assert lut.score_to_quantum(-200) <= lut.score_to_quantum(200)

    # C header export
    c_lut_header = tmp_path / "neuroos_lut.h"
    generate_c_lut_header(lut, str(c_lut_header))
    assert c_lut_header.exists()
    content = c_lut_header.read_text(encoding="utf-8")
    assert "neuroos_quantum_lut[256]" in content
    assert "neuroos_lut_lookup" in content


def test_guardrail_queue_saturation():
    engine = GuardrailEngine(max_queue_depth=1024)
    res_ok = engine.check_guardrails(queue_depth=500)
    assert res_ok.pass_check is True

    res_fail = engine.check_guardrails(queue_depth=1025)
    assert res_fail.pass_check is False
    assert res_fail.reason == FallbackReason.QUEUE_SATURATION
    assert engine.total_trips == 1


def test_guardrail_input_ood():
    engine = GuardrailEngine(feature_upper_bound=10.0)
    normal_feats = np.array([[1.0, 2.0, 0.5] + [0.0] * 13])
    res_ok = engine.check_guardrails(queue_depth=10, candidate_features=normal_feats)
    assert res_ok.pass_check is True

    outlier_feats = np.array([[1.0, 25.0, 0.5] + [0.0] * 13])
    res_fail = engine.check_guardrails(queue_depth=10, candidate_features=outlier_feats)
    assert res_fail.pass_check is False
    assert res_fail.reason == FallbackReason.INPUT_OOD


def test_guardrail_prediction_drift():
    engine = GuardrailEngine(drift_sigma_threshold=3.0)
    # Warmup with errors distributed around 100 with std ~20
    for err in [80, 120, 90, 110, 100, 95, 105, 85, 115, 100] * 3:
        engine.record_prediction_outcome(pred_burst_us=5000, actual_burst_us=5000 + err)

    # Normal residual within 3 sigma
    res_ok = engine.check_guardrails(queue_depth=10, current_running_residual=125.0)
    assert res_ok.pass_check is True

    # Massive drift outlier (> 3 sigma)
    res_fail = engine.check_guardrails(queue_depth=10, current_running_residual=10000.0)
    assert res_fail.pass_check is False
    assert res_fail.reason == FallbackReason.PREDICTION_DRIFT


def test_c_guardrail_check_parity():
    assert c_guardrail_check(queue_depth=500, running_drift_milli_sigma=1000) is True
    assert c_guardrail_check(queue_depth=1025, running_drift_milli_sigma=1000) is False
    assert c_guardrail_check(queue_depth=500, running_drift_milli_sigma=3500) is False


def test_quantization_pipeline_e2e(tmp_path: Path):
    import pytest

    pytest.importorskip("torch")
    from quantization.quantize import run_quantization_pipeline

    checkpoint = Path("ml/checkpoints/student_bc_ppo_s1003.pt")
    if not checkpoint.exists():
        pytest.skip(f"Checkpoint {checkpoint} not committed (gitignored *.pt); skipping e2e.")

    out_dir = tmp_path / "artifacts"
    header_path = tmp_path / "neuroos_weights.h"
    qpol, meta = run_quantization_pipeline(
        checkpoint_path="ml/checkpoints/student_bc_ppo_s1003.pt",
        output_dir=str(out_dir),
        c_header_path=str(header_path),
    )
    assert (out_dir / "quantized_student_int8.json").exists()
    assert (out_dir / "quantized_student_int8.npz").exists()
    assert header_path.exists()
    assert meta["version"] == 1
    assert "scales" in meta
