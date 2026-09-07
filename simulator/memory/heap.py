"""Dynamic Memory Heap Simulation & Fragmentation Tracking."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MemoryBlock:
    offset: int
    size_bytes: int
    is_allocated: bool = False
    task_pid: Optional[int] = None
    predicted_lifetime_us: int = 0
    actual_lifetime_us: int = 0
    requested_size_bytes: int = 0


class SimulatedHeap:
    """
    Simulates contiguous heap memory space and computes external/internal fragmentation.
    Implements formal equations from docs/Metrics.md.
    """
    def __init__(self, total_size_bytes: int = 64 * 1024 * 1024) -> None:
        self.total_size_bytes = total_size_bytes
        self.blocks: List[MemoryBlock] = [
            MemoryBlock(offset=0, size_bytes=total_size_bytes, is_allocated=False)
        ]

    def external_fragmentation(self) -> float:
        """
        External Fragmentation:
        Frag_ext = 1 - (max(Free Block) / sum(Free Blocks))
        """
        free_blocks = [b.size_bytes for b in self.blocks if not b.is_allocated]
        if not free_blocks:
            return 0.0
        sum_free = sum(free_blocks)
        if sum_free == 0:
            return 0.0
        max_free = max(free_blocks)
        return 1.0 - (max_free / sum_free)

    def internal_fragmentation(self) -> float:
        """
        Internal Fragmentation:
        Frag_int = sum(allocated - requested) / sum(allocated)
        """
        allocated_blocks = [b for b in self.blocks if b.is_allocated]
        if not allocated_blocks:
            return 0.0
        total_allocated = sum(b.size_bytes for b in allocated_blocks)
        if total_allocated == 0:
            return 0.0
        total_requested = sum(b.requested_size_bytes for b in allocated_blocks)
        return max(0.0, (total_allocated - total_requested) / total_allocated)

    def buffer_utilization(self) -> float:
        """
        Buffer Utilization:
        eta_buf = sum(requested) / total_heap_bytes
        """
        if self.total_size_bytes == 0:
            return 0.0
        active_requested = sum(b.requested_size_bytes for b in self.blocks if b.is_allocated)
        return active_requested / self.total_size_bytes
