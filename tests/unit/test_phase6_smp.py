"""Unit tests for Phase 6 Multi-Core SMP scheduling and work stealing."""

import unittest
from simulator.scheduling.task import SimulatedTask
from simulator.scheduling.smp_engine import SMPSchedulingSimulationEngine
from schedulers.smp_neuroos.smp_scheduler import SMPNeuroOSLiteScheduler


class TestPhase6SMP(unittest.TestCase):
    def setUp(self):
        # Create a synthetic workload of 20 tasks with varying burst lengths
        self.workload = [
            SimulatedTask(pid=i + 1, arrival_time_us=i * 200, total_burst_us=1000 + (i % 5) * 500)
            for i in range(20)
        ]

    def test_smp_multicore_speedup(self):
        # 1-core baseline
        sched_1 = SMPNeuroOSLiteScheduler(num_cpus=1, num_numa_nodes=1)
        engine_1 = SMPSchedulingSimulationEngine(sched_1, num_cpus=1, num_numa_nodes=1)
        completed_1, metrics_1 = engine_1.run(self.workload)

        # 4-core SMP
        sched_4 = SMPNeuroOSLiteScheduler(num_cpus=4, num_numa_nodes=2)
        engine_4 = SMPSchedulingSimulationEngine(sched_4, num_cpus=4, num_numa_nodes=2)
        completed_4, metrics_4 = engine_4.run(self.workload)

        self.assertEqual(len(completed_1), 20)
        self.assertEqual(len(completed_4), 20)
        # Multi-core makespan should be substantially lower than single-core makespan
        self.assertLess(metrics_4.total_makespan_us, metrics_1.total_makespan_us)

    def test_smp_work_stealing(self):
        # Load all tasks initially onto CPU 0 by setting arrival time to 0
        bursty_workload = [
            SimulatedTask(pid=i + 1, arrival_time_us=0, total_burst_us=3000)
            for i in range(12)
        ]
        sched = SMPNeuroOSLiteScheduler(num_cpus=4, num_numa_nodes=2)
        engine = SMPSchedulingSimulationEngine(sched, num_cpus=4, num_numa_nodes=2)
        completed, metrics = engine.run(bursty_workload)

        self.assertEqual(len(completed), 12)
        # All CPUs should have handled some work
        busy_cores = [b for b in metrics.per_cpu_busy_time_us if b > 0]
        self.assertGreater(len(busy_cores), 1)

    def test_smp_numa_distance(self):
        engine = SMPSchedulingSimulationEngine(None, num_cpus=4, num_numa_nodes=2)
        # CPU 0 and CPU 1 are on node 0
        self.assertEqual(engine.get_numa_distance(0, 0), 10)
        self.assertEqual(engine.get_numa_distance(0, 1), 12)
        # CPU 0 and CPU 2 are on different nodes
        self.assertEqual(engine.get_numa_distance(0, 2), 20)


if __name__ == "__main__":
    unittest.main()
