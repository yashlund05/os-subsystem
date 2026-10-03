"""Unit tests for Phase 6 Online Drift Detection and Self-Correction."""

import unittest
from userspace.drift.online_corrector import OnlineDriftCorrector
from simulator.scheduling.task import SimulatedTask
from simulator.scheduling.smp_engine import SMPSchedulingSimulationEngine
from schedulers.smp_neuroos.smp_scheduler import SMPNeuroOSLiteScheduler


class TestPhase6OnlineDrift(unittest.TestCase):
    def test_corrector_bias_adaptation(self):
        corrector = OnlineDriftCorrector(learning_rate=0.2)
        # Simulate systematic underprediction: model predicts 1000 us, actual is 3000 us
        for _ in range(25):
            corrector.update(predicted_burst_us=1000, actual_burst_us=3000)

        stats = corrector.get_stats()
        self.assertGreater(stats.running_bias_offset, 1000.0)
        # Residual drift sigma should be stabilized
        self.assertLess(stats.current_drift_sigma, 3.0)

    def test_drift_correction_prevents_guardrail_trip(self):
        # Workload with sudden distribution shift
        shifted_workload = [
            SimulatedTask(pid=i + 1, arrival_time_us=i * 100, total_burst_us=8000)
            for i in range(15)
        ]

        # Sched with online drift correction enabled
        sched_with_corr = SMPNeuroOSLiteScheduler(
            num_cpus=2,
            enable_online_drift_correction=True,
            drift_sigma_threshold=3.0,
        )
        engine = SMPSchedulingSimulationEngine(sched_with_corr, num_cpus=2)
        engine.run(shifted_workload)

        # Trips should be zero or negligible
        self.assertEqual(sched_with_corr.fallback_trips, 0)


if __name__ == "__main__":
    unittest.main()
