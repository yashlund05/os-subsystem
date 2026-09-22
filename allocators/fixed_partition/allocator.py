"""Multiprogramming with Fixed Tasks (MFT) Fixed Partition Allocator."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from allocators.base import AllocatorMetrics, BaseAllocator, MemoryHandle


@dataclass
class FixedSlot:
    slot_id: int
    offset: int
    size_bytes: int
    is_allocated: bool = False
    handle: Optional[MemoryHandle] = None


class FixedPartitionAllocator(BaseAllocator):
    """
    Fixed Partitioning Memory Allocator (MFT).
    - Category: Static partition table.
    - Behavior: Heap partitioned into fixed-sized slots.
    - Pathology: High internal fragmentation for small requests; unable to service requests > slot size.
    """

    def __init__(
        self, total_heap_bytes: int = 64 * 1024 * 1024, slot_size_bytes: int = 64 * 1024
    ) -> None:
        super().__init__(name="FixedPartition(MFT)", total_heap_bytes=total_heap_bytes)
        self.slot_size_bytes = slot_size_bytes
        self.num_slots = total_heap_bytes // slot_size_bytes
        self.slots: List[FixedSlot] = [
            FixedSlot(slot_id=i, offset=i * slot_size_bytes, size_bytes=slot_size_bytes)
            for i in range(self.num_slots)
        ]
        self.allocated_handles: Dict[int, FixedSlot] = {}

    def allocate(
        self, request_id: int, task_pid: int, size_bytes: int, predicted_lifetime_us: int = 0
    ) -> Optional[MemoryHandle]:
        self.total_alloc_requests += 1

        if size_bytes > self.slot_size_bytes or size_bytes <= 0:
            self.failed_alloc_requests += 1
            return None

        # Find first free slot
        for slot in self.slots:
            if not slot.is_allocated:
                handle = MemoryHandle(
                    block_id=slot.slot_id,
                    offset=slot.offset,
                    size_bytes=self.slot_size_bytes,
                    requested_size_bytes=size_bytes,
                    task_pid=task_pid,
                    predicted_lifetime_us=predicted_lifetime_us,
                )
                slot.is_allocated = True
                slot.handle = handle
                self.allocated_handles[handle.block_id] = slot
                return handle

        self.failed_alloc_requests += 1
        return None

    def deallocate(self, handle: MemoryHandle) -> bool:
        self.total_dealloc_requests += 1
        slot = self.allocated_handles.pop(handle.block_id, None)
        if slot is None:
            return False
        slot.is_allocated = False
        slot.handle = None
        return True

    def get_metrics(self) -> AllocatorMetrics:
        allocated_slots = [s for s in self.slots if s.is_allocated]
        free_slots = [s for s in self.slots if not s.is_allocated]

        allocated_bytes = len(allocated_slots) * self.slot_size_bytes
        free_bytes = len(free_slots) * self.slot_size_bytes
        max_free_bytes = self.slot_size_bytes if free_slots else 0

        # Internal fragmentation: sum(allocated - requested) / sum(allocated)
        if allocated_slots:
            total_req = sum(s.handle.requested_size_bytes for s in allocated_slots if s.handle)
            int_frag = max(0.0, (allocated_bytes - total_req) / allocated_bytes)
            buf_util = total_req / self.total_heap_bytes
        else:
            int_frag = 0.0
            buf_util = 0.0

        # External fragmentation: 1 - (max_free / sum_free)
        if free_bytes > 0:
            ext_frag = 1.0 - (max_free_bytes / free_bytes)
        else:
            ext_frag = 0.0

        return AllocatorMetrics(
            total_heap_bytes=self.total_heap_bytes,
            allocated_bytes=allocated_bytes,
            free_bytes=free_bytes,
            max_free_block_bytes=max_free_bytes,
            external_fragmentation=ext_frag,
            internal_fragmentation=int_frag,
            buffer_utilization=buf_util,
            total_alloc_requests=self.total_alloc_requests,
            failed_alloc_requests=self.failed_alloc_requests,
            total_dealloc_requests=self.total_dealloc_requests,
        )

    def reset(self) -> None:
        for slot in self.slots:
            slot.is_allocated = False
            slot.handle = None
        self.allocated_handles.clear()
        self.total_alloc_requests = 0
        self.failed_alloc_requests = 0
        self.total_dealloc_requests = 0
