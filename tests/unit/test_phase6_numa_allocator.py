"""Unit tests for Phase 6 NUMA-aware lifetime affinity memory allocator."""

import unittest
from allocators.numa_lifetime.numa_allocator import NUMALifetimeAllocator


class TestPhase6NUMAAllocator(unittest.TestCase):
    def setUp(self):
        self.allocator = NUMALifetimeAllocator(total_heap_bytes=1024 * 1024, num_nodes=2)

    def test_local_node_allocation_preference(self):
        # Allocation targeting node 0
        h0 = self.allocator.allocate(request_id=1, task_pid=10, size_bytes=1024, predicted_lifetime_us=3000, preferred_node=0)
        self.assertIsNotNone(h0)
        self.assertGreaterEqual(self.allocator.local_node_allocations, 1)

        # Allocation targeting node 1
        h1 = self.allocator.allocate(request_id=2, task_pid=11, size_bytes=1024, predicted_lifetime_us=3000, preferred_node=1)
        self.assertIsNotNone(h1)
        self.assertGreaterEqual(self.allocator.local_node_allocations, 2)

        # Offsets should reflect node separation (node 1 starts at half total heap)
        self.assertLess(h0.offset, 512 * 1024)
        self.assertGreaterEqual(h1.offset, 512 * 1024)

    def test_deallocation_and_metrics(self):
        h = self.allocator.allocate(request_id=1, task_pid=1, size_bytes=4096, predicted_lifetime_us=10000, preferred_node=0)
        self.assertIsNotNone(h)
        metrics_before = self.allocator.compute_metrics()
        self.assertGreater(metrics_before.allocated_bytes, 0)

        success = self.allocator.deallocate(h)
        self.assertTrue(success)
        metrics_after = self.allocator.compute_metrics()
        self.assertEqual(metrics_after.allocated_bytes, 0)

    def test_fallback_when_saturated(self):
        # Allocate oversized chunk larger than single band
        huge_h = self.allocator.allocate(request_id=99, task_pid=99, size_bytes=600 * 1024, predicted_lifetime_us=1000, preferred_node=0)
        # Should fall back to buddy allocator
        self.assertIsNotNone(huge_h)
        self.assertGreaterEqual(self.allocator.fallback_trips, 1)


if __name__ == "__main__":
    unittest.main()
