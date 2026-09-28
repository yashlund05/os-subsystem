import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from schedulers.mlfq.scheduler import MLFQScheduler
from schedulers.round_robin.scheduler import RoundRobinScheduler
from simulator.scheduling.engine import SchedulingSimulationEngine
from simulator.scheduling.task import SimulatedTask

def make_tasks():
    return [
        SimulatedTask(pid=1, arrival_time_us=0, total_burst_us=25000),   # Long job
        SimulatedTask(pid=2, arrival_time_us=1000, total_burst_us=2000), # Short interactive job
        SimulatedTask(pid=3, arrival_time_us=2000, total_burst_us=3000), # Short interactive job
    ]

# 1. Run MLFQ through Phase 1 Engine
tasks_m = make_tasks()
engine_mlfq = SchedulingSimulationEngine(
    scheduler=MLFQScheduler(num_levels=3, base_quantum_us=5000, boost_interval_us=500000),
    context_switch_overhead_us=0
)
completed_m, metrics_mlfq = engine_mlfq.run(tasks_m)

# 2. Run RR through Phase 1 Engine
tasks_r = make_tasks()
engine_rr = SchedulingSimulationEngine(
    scheduler=RoundRobinScheduler(quantum_us=5000),
    context_switch_overhead_us=0
)
completed_r, metrics_rr = engine_rr.run(tasks_r)

print("=== Phase 1 SchedulingSimulationEngine: MLFQScheduler vs RoundRobinScheduler ===")
print(f"{'PID':<5} | {'True Burst':<12} | {'MLFQ Completion':<18} | {'RR Completion':<18} | {'Delta (RR - MLFQ)':<18}")
print("-" * 75)
for p in [1, 2, 3]:
    c_m = next(t.completion_time_us for t in completed_m if t.pid == p)
    c_r = next(t.completion_time_us for t in completed_r if t.pid == p)
    b = next(t.total_burst_us for t in tasks_m if t.pid == p)
    print(f"{p:<5} | {b:<12} | {c_m:<18} | {c_r:<18} | {c_r - c_m:<18}")

print(f"\nMean Waiting Time: MLFQ={metrics_mlfq.mean_waiting_time_us:.1f} us vs RR={metrics_rr.mean_waiting_time_us:.1f} us")
print(f"P99 Waiting Time:  MLFQ={metrics_mlfq.p99_waiting_time_us:.1f} us vs RR={metrics_rr.p99_waiting_time_us:.1f} us")
