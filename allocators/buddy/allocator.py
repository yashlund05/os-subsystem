"""Binary Buddy Memory Allocator."""

import math
from typing import Dict, Optional, Set

from allocators.base import AllocatorMetrics, BaseAllocator, MemoryHandle


class BuddyAllocator(BaseAllocator):
    """
    Binary Buddy Memory Allocator.
    - Category: Power-of-two free lists.
    - Sizing: Block sizes are strictly powers of two from min_block_size to total_heap_bytes.
    - Splitting: Larger blocks are recursively split in half (order j -> j-1) until minimal order is reached.
    - Coalescing: When a block is freed, if its buddy (offset ^ size) is also free, they are recursively merged.
    - Pathology: Internal fragmentation up to 49.9% for request sizes equal to 2^k + 1 bytes.
    """

    def __init__(self, total_heap_bytes: int = 64 * 1024 * 1024, min_block_size: int = 64) -> None:
        super().__init__(name="BuddyAllocator", total_heap_bytes=total_heap_bytes)
        self.min_block_size = min_block_size

        # Validate power of two
        self.max_order = int(math.ceil(math.log2(total_heap_bytes)))
        self.min_order = int(math.ceil(math.log2(min_block_size)))
        self.total_heap_bytes = 1 << self.max_order

        # Free lists: dict mapping order -> set of block offsets
        self.free_lists: Dict[int, Set[int]] = {
            order: set() for order in range(self.min_order, self.max_order + 1)
        }
        # Initially, the entire heap is a single free block of max_order
        self.free_lists[self.max_order].add(0)

        # Map block_id -> (offset, order, handle)
        self._allocated_blocks: Dict[int, MemoryHandle] = {}
        self._block_orders: Dict[int, int] = {}  # offset -> order
        self._next_block_id = 1

    def _get_target_order(self, size_bytes: int) -> int:
        req = max(size_bytes, self.min_block_size)
        return max(self.min_order, int(math.ceil(math.log2(req))))

    def allocate(
        self, request_id: int, task_pid: int, size_bytes: int, predicted_lifetime_us: int = 0
    ) -> Optional[MemoryHandle]:
        self.total_alloc_requests += 1
        if size_bytes <= 0 or size_bytes > self.total_heap_bytes:
            self.failed_alloc_requests += 1
            return None

        target_order = self._get_target_order(size_bytes)

        # Find smallest available order >= target_order
        found_order: Optional[int] = None
        for order in range(target_order, self.max_order + 1):
            if self.free_lists[order]:
                found_order = order
                break

        if found_order is None:
            self.failed_alloc_requests += 1
            return None

        # Remove block from free list
        block_offset = min(self.free_lists[found_order])
        self.free_lists[found_order].remove(block_offset)

        # Recursively split down to target_order
        while found_order > target_order:
            found_order -= 1
            buddy_offset = block_offset + (1 << found_order)
            self.free_lists[found_order].add(buddy_offset)

        # Allocate block
        allocated_size = 1 << target_order
        handle = MemoryHandle(
            block_id=self._next_block_id,
            offset=block_offset,
            size_bytes=allocated_size,
            requested_size_bytes=size_bytes,
            task_pid=task_pid,
            predicted_lifetime_us=predicted_lifetime_us,
        )
        self._next_block_id += 1
        self._allocated_blocks[handle.block_id] = handle
        self._block_orders[block_offset] = target_order

        return handle

    def deallocate(self, handle: MemoryHandle) -> bool:
        self.total_dealloc_requests += 1
        stored_handle = self._allocated_blocks.pop(handle.block_id, None)
        if stored_handle is None:
            return False

        offset = stored_handle.offset
        order = self._block_orders.pop(offset, None)
        if order is None:
            return False

        # Recursively coalesce with buddy if buddy is free
        while order < self.max_order:
            buddy_offset = offset ^ (1 << order)
            if buddy_offset in self.free_lists[order]:
                # Remove buddy from free list and merge
                self.free_lists[order].remove(buddy_offset)
                offset = min(offset, buddy_offset)
                order += 1
            else:
                break

        self.free_lists[order].add(offset)
        return True

    def get_metrics(self) -> AllocatorMetrics:
        free_bytes = sum(len(offsets) * (1 << order) for order, offsets in self.free_lists.items())
        allocated_bytes = sum(h.size_bytes for h in self._allocated_blocks.values())

        # Find maximum free block size
        max_free_bytes = 0
        for order in range(self.max_order, self.min_order - 1, -1):
            if self.free_lists[order]:
                max_free_bytes = 1 << order
                break

        # External fragmentation: 1 - (max_free / sum_free)
        if free_bytes > 0:
            ext_frag = 1.0 - (max_free_bytes / free_bytes)
        else:
            ext_frag = 0.0

        # Internal fragmentation: sum(allocated - requested) / sum(allocated)
        if allocated_bytes > 0:
            total_req = sum(h.requested_size_bytes for h in self._allocated_blocks.values())
            int_frag = max(0.0, (allocated_bytes - total_req) / allocated_bytes)
            buf_util = total_req / self.total_heap_bytes
        else:
            int_frag = 0.0
            buf_util = 0.0

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
        self.free_lists = {order: set() for order in range(self.min_order, self.max_order + 1)}
        self.free_lists[self.max_order].add(0)
        self._allocated_blocks.clear()
        self._block_orders.clear()
        self._next_block_id = 1
        self.total_alloc_requests = 0
        self.failed_alloc_requests = 0
        self.total_dealloc_requests = 0
