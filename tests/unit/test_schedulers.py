"""Unit tests for the 5 classical CPU scheduling algorithms."""

from schedulers.fcfs.scheduler import FCFSScheduler
from schedulers.mlfq.scheduler import MLFQScheduler
from schedulers.round_robin.scheduler import RoundRobinScheduler
from schedulers.sjf.scheduler import SJFScheduler
from schedulers.srtf.scheduler import SRTFScheduler
from simulator.scheduling.task import SimulatedTask


def test_fcfs_scheduler():
    sched = FCFSScheduler()
    assert not sched.is_preemptive
    assert not sched.has_runnable_tasks()

    t1 = SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=100)
    t2 = SimulatedTask(pid=2, arrival_time_us=5, total_burst_us=50)

    sched.add_task(t1, 0)
    sched.add_task(t2, 5)

    assert sched.get_queue_depth() == 2

    # FCFS should return t1 first, granting full remaining burst
    picked, q = sched.pick_next_task(5)
    assert picked is not None and picked.pid == 1
    assert q == 100

    picked2, q2 = sched.pick_next_task(105)
    assert picked2 is not None and picked2.pid == 2
    assert q2 == 50

    picked3, q3 = sched.pick_next_task(155)
    assert picked3 is None and q3 == 0


def test_sjf_scheduler():
    sched = SJFScheduler()
    assert not sched.is_preemptive

    t1 = SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=200)
    t2 = SimulatedTask(pid=2, arrival_time_us=0, total_burst_us=50)
    t3 = SimulatedTask(pid=3, arrival_time_us=0, total_burst_us=100)

    sched.add_task(t1, 0)
    sched.add_task(t2, 0)
    sched.add_task(t3, 0)

    # SJF should pick shortest total burst: t2 (50), then t3 (100), then t1 (200)
    p1, _ = sched.pick_next_task(0)
    p2, _ = sched.pick_next_task(50)
    p3, _ = sched.pick_next_task(150)

    assert p1 is not None and p1.pid == 2
    assert p2 is not None and p2.pid == 3
    assert p3 is not None and p3.pid == 1


def test_srtf_scheduler():
    sched = SRTFScheduler()
    assert sched.is_preemptive

    t1 = SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=200)
    t1.remaining_burst_us = 150  # Executed 50 us

    t2 = SimulatedTask(pid=2, arrival_time_us=50, total_burst_us=80)

    # t2 remaining burst (80) < t1 remaining burst (150) -> should preempt
    assert sched.should_preempt(t1, t2)

    t3 = SimulatedTask(pid=3, arrival_time_us=60, total_burst_us=300)
    # t3 remaining burst (300) > t1 remaining burst (150) -> should not preempt
    assert not sched.should_preempt(t1, t3)

    sched.add_task(t1, 50)
    sched.add_task(t2, 50)

    # Pick next should yield t2 because 80 < 150
    picked, _ = sched.pick_next_task(50)
    assert picked is not None and picked.pid == 2


def test_round_robin_scheduler():
    sched = RoundRobinScheduler(quantum_us=20)
    assert sched.is_preemptive

    t1 = SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=50)
    t2 = SimulatedTask(pid=2, arrival_time_us=0, total_burst_us=30)

    sched.add_task(t1, 0)
    sched.add_task(t2, 0)

    # Turn 1: t1 gets 20 us
    p1, q1 = sched.pick_next_task(0)
    assert p1.pid == 1 and q1 == 20
    p1.remaining_burst_us -= 20
    sched.on_task_preempted(p1, 20)

    # Turn 2: t2 gets 20 us
    p2, q2 = sched.pick_next_task(20)
    assert p2.pid == 2 and q2 == 20
    p2.remaining_burst_us -= 20
    sched.on_task_preempted(p2, 40)

    # Turn 3: t1 gets 20 us (remaining burst is 30)
    p3, q3 = sched.pick_next_task(40)
    assert p3.pid == 1 and q3 == 20


def test_mlfq_scheduler_demotion_and_boost():
    sched = MLFQScheduler(num_levels=3, base_quantum_us=10, boost_interval_us=100)
    assert sched.is_preemptive

    t1 = SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=100)
    sched.add_task(t1, 0)

    # Initially at level 0, gets quantum = 10
    p1, q1 = sched.pick_next_task(0)
    assert p1.pid == 1 and q1 == 10
    assert sched.task_levels[1] == 0

    # Task uses full quantum, preempted -> demoted to level 1
    p1.remaining_burst_us -= 10
    sched.on_task_preempted(p1, 10)
    assert sched.task_levels[1] == 1

    # Level 1 quantum should be base * 2^1 = 20
    p2, q2 = sched.pick_next_task(10)
    assert p2.pid == 1 and q2 == 20

    # Demoted to level 2
    p2.remaining_burst_us -= 20
    sched.on_task_preempted(p2, 30)
    assert sched.task_levels[1] == 2

    # Advance time to trigger priority boost (> 100 us)
    sched.on_tick(current_time_us=105, elapsed_us=10)
    # Pick next should now be back at level 0 with base quantum 10
    p3, q3 = sched.pick_next_task(105)
    assert p3.pid == 1 and q3 == 10
    assert sched.task_levels[1] == 0
