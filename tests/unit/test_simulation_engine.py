"""Unit tests for the discrete-event CPU scheduling simulation engine."""

from schedulers.fcfs.scheduler import FCFSScheduler
from schedulers.mlfq.scheduler import MLFQScheduler
from schedulers.round_robin.scheduler import RoundRobinScheduler
from schedulers.sjf.scheduler import SJFScheduler
from schedulers.srtf.scheduler import SRTFScheduler
from simulator.scheduling.engine import SchedulingSimulationEngine
from simulator.scheduling.task import SimulatedTask, TaskState


def test_simulation_engine_fcfs():
    workload = [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=100),
        SimulatedTask(pid=2, arrival_time_us=10, total_burst_us=50),
        SimulatedTask(pid=3, arrival_time_us=20, total_burst_us=80),
    ]

    sched = FCFSScheduler()
    engine = SchedulingSimulationEngine(scheduler=sched, context_switch_overhead_us=2)
    completed, metrics = engine.run(workload)

    assert len(completed) == 3
    assert all(t.state == TaskState.COMPLETED for t in completed)

    # Invariants
    total_burst = sum(t.total_burst_us for t in workload)
    assert metrics.total_cpu_busy_time_us == total_burst
    assert metrics.completed_tasks == 3
    assert metrics.total_context_switches >= 2  # between t1->t2 and t2->t3

    for t in completed:
        assert t.turnaround_time_us is not None
        assert t.turnaround_time_us >= t.total_burst_us
        assert t.normalized_turnaround_time is not None
        assert t.normalized_turnaround_time >= 1.0


def test_simulation_engine_sjf_vs_srtf():
    # Job 1 arrives at t=0 with burst 100.
    # Job 2 arrives at t=10 with burst 20.
    # Under SJF (non-preemptive), Job 1 runs to completion (100 us), Job 2 must wait until t=100.
    # Under SRTF (preemptive), Job 2 arrives at t=10, preempts Job 1 (remaining 90 > 20), finishes at t=30.
    workload_sjf = [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=100),
        SimulatedTask(pid=2, arrival_time_us=10, total_burst_us=20),
    ]
    workload_srtf = [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=100),
        SimulatedTask(pid=2, arrival_time_us=10, total_burst_us=20),
    ]

    engine_sjf = SchedulingSimulationEngine(scheduler=SJFScheduler(), context_switch_overhead_us=0)
    completed_sjf, metrics_sjf = engine_sjf.run(workload_sjf)

    engine_srtf = SchedulingSimulationEngine(
        scheduler=SRTFScheduler(), context_switch_overhead_us=0
    )
    completed_srtf, metrics_srtf = engine_srtf.run(workload_srtf)

    task2_sjf = next(t for t in completed_sjf if t.pid == 2)
    task2_srtf = next(t for t in completed_srtf if t.pid == 2)

    # Under SJF: task 2 completion time = 120 us.
    # Under SRTF: task 2 completion time = 30 us.
    assert task2_sjf.completion_time_us == 120
    assert task2_srtf.completion_time_us == 30
    assert metrics_srtf.mean_waiting_time_us < metrics_sjf.mean_waiting_time_us


def test_simulation_engine_round_robin_and_mlfq():
    workload = [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=30),
        SimulatedTask(pid=2, arrival_time_us=0, total_burst_us=50),
        SimulatedTask(pid=3, arrival_time_us=0, total_burst_us=20),
    ]

    # Test Round Robin
    rr = RoundRobinScheduler(quantum_us=10)
    engine_rr = SchedulingSimulationEngine(scheduler=rr, context_switch_overhead_us=1)
    completed_rr, metrics_rr = engine_rr.run(workload)
    assert len(completed_rr) == 3
    assert metrics_rr.total_context_switches > 0

    # Test MLFQ
    mlfq = MLFQScheduler(num_levels=3, base_quantum_us=10, boost_interval_us=50)
    engine_mlfq = SchedulingSimulationEngine(scheduler=mlfq, context_switch_overhead_us=1)
    completed_mlfq, metrics_mlfq = engine_mlfq.run(workload)
    assert len(completed_mlfq) == 3
    assert metrics_mlfq.completed_tasks == 3
