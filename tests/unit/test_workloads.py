"""Unit tests for workload generators and benchmark runners."""

import io

from allocators.best_fit.allocator import BestFitAllocator
from benchmarks.runner import BenchmarkRunner
from schedulers.fcfs.scheduler import FCFSScheduler
from simulator.workloads.adversarial import AdversarialWorkloadGenerator
from simulator.workloads.synthetic import SyntheticWorkloadGenerator
from simulator.workloads.trace_parser import ClusterTraceParser


def test_synthetic_pareto_load_factors():
    gen = SyntheticWorkloadGenerator(seed=42)

    # Test load factor 0.20 vs 0.80
    tasks_low = gen.generate_pareto_bursts(
        num_tasks=20, alpha=1.3, min_burst_us=100, load_factor=0.20
    )
    tasks_high = gen.generate_pareto_bursts(
        num_tasks=20, alpha=1.3, min_burst_us=100, load_factor=0.80
    )

    assert len(tasks_low) == 20
    assert len(tasks_high) == 20

    # Lower load factor should have greater average inter-arrival time (larger final arrival time)
    assert tasks_low[-1].arrival_time_us > tasks_high[-1].arrival_time_us


def test_cluster_trace_parser():
    sample_csv = """pid,arrival_us,burst_us,cache_misses,branch_mispred,mem_kb
101,0,500,12,3,2048
102,150,300,5,1,1024
103,100,800,20,8,4096
"""
    stream = io.StringIO(sample_csv)
    tasks = ClusterTraceParser.parse_csv_stream(stream)

    assert len(tasks) == 3
    # Check sorted by arrival time: pid 101 (t=0), pid 103 (t=100), pid 102 (t=150)
    assert tasks[0].pid == 101
    assert tasks[1].pid == 103
    assert tasks[2].pid == 102
    assert tasks[0].total_burst_us == 500
    assert tasks[1].cache_misses == 20


def test_adversarial_convoy_workload():
    convoy = AdversarialWorkloadGenerator.create_convoy_workload(
        num_short_jobs=10, long_burst_us=10000, short_burst_us=10
    )
    assert len(convoy) == 11
    assert convoy[0].pid == 1 and convoy[0].total_burst_us == 10000
    assert convoy[1].pid == 2 and convoy[1].total_burst_us == 10
    assert convoy[-1].pid == 11 and convoy[-1].total_burst_us == 10


def test_adversarial_memory_fragmentation_churn():
    events = AdversarialWorkloadGenerator.create_memory_fragmentation_churn(
        num_pairs=5, small_size_bytes=100, large_size_bytes=500
    )
    # 5 pairs = 10 allocs + 5 frees = 15 events
    assert len(events) == 15
    allocs = [e for e in events if e.is_alloc]
    frees = [e for e in events if not e.is_alloc]
    assert len(allocs) == 10
    assert len(frees) == 5


def test_benchmark_runner():
    # Run a small scheduler benchmark
    sched = FCFSScheduler()
    gen = SyntheticWorkloadGenerator(seed=99)
    workload = gen.generate_pareto_bursts(num_tasks=5, min_burst_us=50)

    res_sched = BenchmarkRunner.run_scheduler_benchmark(
        scheduler=sched, workload=workload, experiment_id="test_exp_sched_01", load_factor=0.5
    )
    assert res_sched.experiment_id == "test_exp_sched_01"
    assert "mean_turnaround_time_us" in res_sched.metrics
    assert res_sched.metrics["total_makespan_us"] > 0

    # Run a small memory benchmark
    alloc = BestFitAllocator(total_heap_bytes=10000)
    events = AdversarialWorkloadGenerator.create_memory_fragmentation_churn(num_pairs=3)
    res_mem = BenchmarkRunner.run_memory_benchmark(
        allocator=alloc, events=events, experiment_id="test_exp_mem_01"
    )
    assert res_mem.experiment_id == "test_exp_mem_01"
    assert "external_fragmentation" in res_mem.metrics
