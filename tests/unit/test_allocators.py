"""Unit tests for the 4 classical dynamic memory allocators."""

import pytest

from allocators.best_fit.allocator import BestFitAllocator
from allocators.buddy.allocator import BuddyAllocator
from allocators.first_fit.allocator import FirstFitAllocator
from allocators.fixed_partition.allocator import FixedPartitionAllocator


def test_fixed_partition_allocator():
    # 4 slots of 1000 bytes each
    alloc = FixedPartitionAllocator(total_heap_bytes=4000, slot_size_bytes=1000)

    h1 = alloc.allocate(request_id=1, task_pid=10, size_bytes=600)
    assert h1 is not None and h1.size_bytes == 1000 and h1.requested_size_bytes == 600

    # Oversized request fails
    h_oversized = alloc.allocate(request_id=2, task_pid=11, size_bytes=1500)
    assert h_oversized is None

    metrics = alloc.get_metrics()
    assert metrics.allocated_bytes == 1000
    assert metrics.free_bytes == 3000
    # Internal fragmentation: (1000 - 600) / 1000 = 0.40
    assert pytest.approx(metrics.internal_fragmentation, 0.001) == 0.40

    # Free slot
    assert alloc.deallocate(h1)
    metrics_after = alloc.get_metrics()
    assert metrics_after.allocated_bytes == 0
    assert metrics_after.free_bytes == 4000


def test_first_fit_allocator_coalescing():
    alloc = FirstFitAllocator(total_heap_bytes=1000, min_split_bytes=50)

    h1 = alloc.allocate(request_id=1, task_pid=1, size_bytes=200)
    h2 = alloc.allocate(request_id=2, task_pid=2, size_bytes=300)
    h3 = alloc.allocate(request_id=3, task_pid=3, size_bytes=200)

    assert h1 is not None and h2 is not None and h3 is not None
    assert h1.offset == 0
    assert h2.offset == 200
    assert h3.offset == 500

    # Deallocate h2: leaves a 300-byte hole at offset 200
    assert alloc.deallocate(h2)

    # Next allocation of 150 bytes should fit in the hole at offset 200 (First-Fit)
    h4 = alloc.allocate(request_id=4, task_pid=4, size_bytes=150)
    assert h4 is not None and h4.offset == 200

    # Clean up and verify full coalescing back to 1000 bytes
    assert alloc.deallocate(h1)
    assert alloc.deallocate(h4)
    assert alloc.deallocate(h3)

    metrics = alloc.get_metrics()
    assert metrics.free_bytes == 1000
    assert metrics.max_free_block_bytes == 1000
    assert metrics.external_fragmentation == 0.0


def test_best_fit_allocator_selection():
    alloc = BestFitAllocator(total_heap_bytes=1000, min_split_bytes=50)

    # Allocate 3 blocks: 200, 300, 200
    h1 = alloc.allocate(request_id=1, task_pid=1, size_bytes=200)
    h2 = alloc.allocate(request_id=2, task_pid=2, size_bytes=300)
    h3 = alloc.allocate(request_id=3, task_pid=3, size_bytes=200)
    assert h1 is not None and h2 is not None and h3 is not None

    # Free h1 (200 B at offset 0) and h2 (300 B at offset 200)
    alloc.deallocate(h1)
    alloc.deallocate(h2)
    # Note: h1 and h2 are adjacent, so they coalesced into a 500-byte block.
    # Let's recreate isolated holes:
    alloc.reset()
    h1 = alloc.allocate(request_id=1, task_pid=1, size_bytes=100)  # 0..100
    s1 = alloc.allocate(request_id=2, task_pid=2, size_bytes=100)  # 100..200
    h2 = alloc.allocate(request_id=3, task_pid=3, size_bytes=300)  # 200..500
    s2 = alloc.allocate(request_id=4, task_pid=4, size_bytes=100)  # 500..600
    assert s1 is not None and s2 is not None

    # Free h1 (100 bytes hole) and h2 (300 bytes hole)
    alloc.deallocate(h1)
    alloc.deallocate(h2)

    # Now request 80 bytes:
    # First-fit would check 100 B hole first (fits, excess 20).
    # Best-fit should choose the 100 B hole because excess 20 < excess 220!
    h_small = alloc.allocate(request_id=5, task_pid=5, size_bytes=80)
    assert h_small is not None and h_small.offset == 0


def test_buddy_allocator_splitting_and_coalescing():
    # Heap of 1024 bytes (power of 2), min block 64 bytes
    alloc = BuddyAllocator(total_heap_bytes=1024, min_block_size=64)

    # Request 100 bytes -> rounds up to 128 bytes (2^7)
    h1 = alloc.allocate(request_id=1, task_pid=1, size_bytes=100)
    assert h1 is not None and h1.size_bytes == 128

    # Internal fragmentation: (128 - 100) / 128 = 28 / 128 = 0.21875
    metrics = alloc.get_metrics()
    assert pytest.approx(metrics.internal_fragmentation, 0.001) == 28.0 / 128.0

    # Free h1 and verify buddy coalescing restores entire 1024-byte block
    assert alloc.deallocate(h1)
    metrics_free = alloc.get_metrics()
    assert metrics_free.free_bytes == 1024
    assert metrics_free.max_free_block_bytes == 1024
    assert metrics_free.external_fragmentation == 0.0
