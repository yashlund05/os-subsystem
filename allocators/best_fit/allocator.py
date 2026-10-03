"""Dynamic Variable Partitioning Best-Fit Memory Allocator."""

from typing import Dict, List, Optional

from allocators.base import AllocatorMetrics, BaseAllocator, MemoryHandle
from allocators.first_fit.allocator import HeapBlock


class BestFitAllocator(BaseAllocator):
    """
    Dynamic Best-Fit Memory Allocator.
    - Category: Variable partitioning (MVT).
    - Policy: Searches entire free list to find the block that minimizes excess size (smallest viable block).
    - Splitting: Splits block if excess >= min_split_bytes.
    - Coalescing: Merges adjacent free blocks on deallocation.
    - Pathology: Leaves behind tiny, unusable memory holes across the address space (severe external fragmentation).
    """

    def __init__(self, total_heap_bytes: int = 64 * 1024 * 1024, min_split_bytes: int = 64) -> None:
        super().__init__(name="BestFit", total_heap_bytes=total_heap_bytes)
        self.min_split_bytes = min_split_bytes
        self._next_block_id = 1
        self.blocks: List[HeapBlock] = [
            HeapBlock(
                block_id=self._next_block_id,
                offset=0,
                size_bytes=total_heap_bytes,
                is_allocated=False,
            )
        ]
        self._allocated_handles: Dict[int, HeapBlock] = {}

    def allocate(
        self, request_id: int, task_pid: int, size_bytes: int, predicted_lifetime_us: int = 0
    ) -> Optional[MemoryHandle]:
        self.total_alloc_requests += 1
        if size_bytes <= 0:
            self.failed_alloc_requests += 1
            return None

        # Search for best fitting block (minimum excess >= 0)
        best_idx: Optional[int] = None
        min_excess = float("inf")

        for idx, block in enumerate(self.blocks):
            if not block.is_allocated and block.size_bytes >= size_bytes:
                excess = block.size_bytes - size_bytes
                if excess < min_excess:
                    min_excess = excess
                    best_idx = idx

        if best_idx is None:
            self.failed_alloc_requests += 1
            return None

        block = self.blocks[best_idx]
        excess = block.size_bytes - size_bytes

        if excess >= self.min_split_bytes:
            # Split block
            self._next_block_id += 1
            new_free_block = HeapBlock(
                block_id=self._next_block_id,
                offset=block.offset + size_bytes,
                size_bytes=excess,
                is_allocated=False,
            )
            block.size_bytes = size_bytes
            self.blocks.insert(best_idx + 1, new_free_block)

        block.is_allocated = True
        handle = MemoryHandle(
            block_id=block.block_id,
            offset=block.offset,
            size_bytes=block.size_bytes,
            requested_size_bytes=size_bytes,
            task_pid=task_pid,
            predicted_lifetime_us=predicted_lifetime_us,
        )
        block.handle = handle
        self._allocated_handles[handle.block_id] = block
        return handle

    def deallocate(self, handle: MemoryHandle) -> bool:
        self.total_dealloc_requests += 1
        block = self._allocated_handles.pop(handle.block_id, None)
        if block is None:
            return False

        block.is_allocated = False
        block.handle = None
        self._coalesce()
        return True

    def _coalesce(self) -> None:
        """Merge adjacent free blocks in physical memory order."""
        i = 0
        while i < len(self.blocks) - 1:
            curr_b = self.blocks[i]
            next_b = self.blocks[i + 1]
            if not curr_b.is_allocated and not next_b.is_allocated:
                curr_b.size_bytes += next_b.size_bytes
                self.blocks.pop(i + 1)
            else:
                i += 1

    def get_metrics(self) -> AllocatorMetrics:
        free_blocks = [b.size_bytes for b in self.blocks if not b.is_allocated]
        allocated_blocks = [b for b in self.blocks if b.is_allocated]

        free_bytes = sum(free_blocks)
        allocated_bytes = sum(b.size_bytes for b in allocated_blocks)
        max_free_bytes = max(free_blocks) if free_blocks else 0

        # External fragmentation: 1 - (max_free / sum_free)
        if free_bytes > 0:
            ext_frag = 1.0 - (max_free_bytes / free_bytes)
        else:
            ext_frag = 0.0

        # Internal fragmentation: sum(allocated - requested) / sum(allocated)
        if allocated_bytes > 0:
            total_req = sum(b.handle.requested_size_bytes for b in allocated_blocks if b.handle)
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
        self._next_block_id = 1
        self.blocks = [
            HeapBlock(
                block_id=self._next_block_id,
                offset=0,
                size_bytes=self.total_heap_bytes,
                is_allocated=False,
            )
        ]
        self._allocated_handles.clear()
        self.total_alloc_requests = 0
        self.failed_alloc_requests = 0
        self.total_dealloc_requests = 0
