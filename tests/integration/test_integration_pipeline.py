"""Integration tests – Phase 3, Week 5.

Covers:
1. SPSC ring-buffer kernel-producer / userspace-consumer data-flow.
2. Scheduler queue dispatch and dynamic quantum expiration loop.
3. Guardrail trip and fallback recovery sequences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make repo root importable when running pytest from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# 1. SPSC Ring-Buffer data-flow integration
# ---------------------------------------------------------------------------


def test_ring_buffer_produce_consume_roundtrip() -> None:
    """Events written to the ring buffer can be read back intact (FIFO order)."""
    from telemetry.ring_buffer.ring_buffer import RingBuffer  # type: ignore

    rb = RingBuffer(capacity=16)
    items = [{"pid": i, "cpu_cycles": i * 100} for i in range(8)]
    for item in items:
        assert rb.push(item) is True, "push should succeed for a non-full buffer"

    recovered = []
    while not rb.empty():
        recovered.append(rb.pop())

    assert recovered == items, "Recovered items must match insertion order (FIFO)"


def test_ring_buffer_wraparound_no_data_loss() -> None:
    """Produce and consume across a full wraparound cycle without data loss."""
    from telemetry.ring_buffer.ring_buffer import RingBuffer  # type: ignore

    rb = RingBuffer(capacity=4)
    for i in range(4):
        rb.push({"seq": i})
    # Drain half
    rb.pop()
    rb.pop()
    # Push two more (causes logical wraparound in circular slot indexing)
    rb.push({"seq": 4})
    rb.push({"seq": 5})

    seqs = [rb.pop()["seq"] for _ in range(4)]
    assert seqs == [2, 3, 4, 5], f"Expected [2,3,4,5] after wraparound, got {seqs}"


def test_ring_buffer_full_push_rejected() -> None:
    """A push to a completely full buffer must be rejected without blocking."""
    from telemetry.ring_buffer.ring_buffer import RingBuffer  # type: ignore

    rb = RingBuffer(capacity=4)
    for i in range(4):
        assert rb.push({"v": i}) is True
    assert rb.push({"v": 99}) is False, "push to full buffer must return False"


def test_ring_buffer_empty_pop_returns_none() -> None:
    """Popping from an empty buffer must return None (no blocking)."""
    from telemetry.ring_buffer.ring_buffer import RingBuffer  # type: ignore

    rb = RingBuffer(capacity=8)
    assert rb.pop() is None


def test_ring_buffer_capacity_rounded_to_power_of_two() -> None:
    """Capacity is rounded up to the next power-of-two (mirrors C implementation)."""
    from telemetry.ring_buffer.ring_buffer import RingBuffer  # type: ignore

    rb = RingBuffer(capacity=5)
    assert rb.capacity == 8
    rb2 = RingBuffer(capacity=16)
    assert rb2.capacity == 16


# ---------------------------------------------------------------------------
# 2. Guardrail trip and fallback recovery sequences
# ---------------------------------------------------------------------------


def test_guardrail_trips_on_deep_queue() -> None:
    """Guardrail must activate when queue depth exceeds 1024."""
    from kernel.guardrails.guardrail import Guardrail  # type: ignore

    g = Guardrail(queue_depth_threshold=1024, drift_sigma_threshold=3.0)
    assert g.check(queue_depth=512, prediction_error_sigma=1.0) is False
    assert g.check(queue_depth=1025, prediction_error_sigma=1.0) is True
    assert g.last_reason == "queue_saturation"


def test_guardrail_trips_on_high_drift() -> None:
    """Guardrail must activate when prediction drift exceeds 3 σ."""
    from kernel.guardrails.guardrail import Guardrail  # type: ignore

    g = Guardrail(queue_depth_threshold=1024, drift_sigma_threshold=3.0)
    assert g.check(queue_depth=10, prediction_error_sigma=3.5) is True
    assert g.last_reason == "drift_trip"


def test_guardrail_recovery_after_trip() -> None:
    """After a trip, the guardrail must recover once conditions normalise."""
    from kernel.guardrails.guardrail import Guardrail  # type: ignore

    g = Guardrail(queue_depth_threshold=1024, drift_sigma_threshold=3.0)
    assert g.check(queue_depth=2000, prediction_error_sigma=4.0) is True
    assert g.check(queue_depth=100, prediction_error_sigma=0.5) is False
    assert g.last_reason == "none"


def test_guardrail_boundary_exact_threshold() -> None:
    """Exactly at the threshold (=) must NOT trip (only strictly > triggers)."""
    from kernel.guardrails.guardrail import Guardrail  # type: ignore

    g = Guardrail(queue_depth_threshold=1024, drift_sigma_threshold=3.0)
    assert g.check(queue_depth=1024, prediction_error_sigma=3.0) is False


# ---------------------------------------------------------------------------
# 3. NeuroOS-Lite scheduler dispatch integration
# ---------------------------------------------------------------------------


def _make_tasks(n: int = 10, seed: int = 1):
    """Build a simple list of SimulatedTask objects with deterministic bursts."""
    from simulator.scheduling.task import SimulatedTask  # type: ignore
    import numpy as np

    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        burst_us = int(rng.integers(1000, 20000))
        tasks.append(SimulatedTask(pid=i + 1, arrival_time_us=i * 200, total_burst_us=burst_us))
    return tasks


def test_neuroos_scheduler_dispatches_all_tasks() -> None:
    """NeuroOS-Lite scheduler must dispatch every submitted task to completion."""
    from schedulers.neuroos_lite.scheduler import NeuroOSLiteScheduler  # type: ignore

    sched = NeuroOSLiteScheduler()
    tasks = _make_tasks(n=20, seed=42)
    current_time = 0
    for t in tasks:
        sched.add_task(t, t.arrival_time_us)

    dispatched = 0
    max_iters = len(tasks) * 500
    for _ in range(max_iters):
        if not sched.has_runnable_tasks():
            break
        task, quantum = sched.pick_next_task(current_time)
        if task is None:
            break
        # Simulate running the task for its quantum or remaining burst
        run_for = min(quantum, task.remaining_burst_us)
        task.remaining_burst_us -= run_for
        current_time += run_for
        if task.remaining_burst_us <= 0:
            sched.on_task_completion(task, current_time)
            dispatched += 1
        else:
            sched.on_task_preempted(task, current_time)

    assert dispatched == len(tasks), (
        f"Only {dispatched}/{len(tasks)} tasks completed after {max_iters} iterations"
    )


def test_neuroos_scheduler_guardrail_fallback_recorded() -> None:
    """NeuroOS-Lite must record fallback trips when queue depth exceeds 1024."""
    from schedulers.neuroos_lite.scheduler import NeuroOSLiteScheduler  # type: ignore

    # Use a very low guardrail threshold to trigger easily
    sched = NeuroOSLiteScheduler(max_queue_depth=2)
    tasks = _make_tasks(n=10, seed=77)
    current_time = 0
    for t in tasks:
        sched.add_task(t, t.arrival_time_us)

    # Pump until all tasks complete or guardrail trips recorded
    for _ in range(len(tasks) * 500):
        if not sched.has_runnable_tasks():
            break
        task, quantum = sched.pick_next_task(current_time)
        if task is None:
            break
        run_for = min(quantum, task.remaining_burst_us)
        task.remaining_burst_us -= run_for
        current_time += run_for
        if task.remaining_burst_us <= 0:
            sched.on_task_completion(task, current_time)
        else:
            sched.on_task_preempted(task, current_time)

    assert sched.fallback_trips >= 1, (
        "Expected ≥1 guardrail fallback trip with max_queue_depth=2 and 10 tasks"
    )


def test_mlfq_baseline_dispatches_all_tasks() -> None:
    """MLFQ baseline scheduler must complete every submitted task."""
    from schedulers.mlfq.scheduler import MLFQScheduler  # type: ignore

    sched = MLFQScheduler()
    tasks = _make_tasks(n=15, seed=5)
    current_time = 0
    for t in tasks:
        sched.add_task(t, t.arrival_time_us)

    dispatched = 0
    for _ in range(len(tasks) * 500):
        if not sched.has_runnable_tasks():
            break
        task, quantum = sched.pick_next_task(current_time)
        if task is None:
            break
        run_for = min(quantum, task.remaining_burst_us)
        task.remaining_burst_us -= run_for
        current_time += run_for
        if task.remaining_burst_us <= 0:
            sched.on_task_completion(task, current_time)
            dispatched += 1
        else:
            sched.on_task_preempted(task, current_time)

    assert dispatched == len(tasks), (
        f"MLFQ: only {dispatched}/{len(tasks)} tasks completed"
    )
