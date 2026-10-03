"""Unified base class for all memory allocator implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MemoryHandle:
    """Represents an allocated memory block handle returned to the caller."""

    block_id: int
    offset: int
    size_bytes: int
    requested_size_bytes: int
    task_pid: int
    predicted_lifetime_us: int = 0


@dataclass
class AllocatorMetrics:
    """Metrics tracking memory usage and fragmentation."""

    total_heap_bytes: int
    allocated_bytes: int
    free_bytes: int
    max_free_block_bytes: int
    external_fragmentation: float
    internal_fragmentation: float
    buffer_utilization: float
    total_alloc_requests: int
    failed_alloc_requests: int
    total_dealloc_requests: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_heap_bytes": float(self.total_heap_bytes),
            "allocated_bytes": float(self.allocated_bytes),
            "free_bytes": float(self.free_bytes),
            "max_free_block_bytes": float(self.max_free_block_bytes),
            "external_fragmentation": self.external_fragmentation,
            "internal_fragmentation": self.internal_fragmentation,
            "buffer_utilization": self.buffer_utilization,
            "total_alloc_requests": float(self.total_alloc_requests),
            "failed_alloc_requests": float(self.failed_alloc_requests),
            "total_dealloc_requests": float(self.total_dealloc_requests),
        }


class BaseAllocator(ABC):
    """
    Abstract base class for memory partition allocators.
    Mirrors `struct allocator_ops` from kernel/allocators/include/allocator_interface.h.
    """

    def __init__(self, name: str, total_heap_bytes: int) -> None:
        self.name = name
        self.total_heap_bytes = total_heap_bytes
        self.total_alloc_requests = 0
        self.failed_alloc_requests = 0
        self.total_dealloc_requests = 0

    @abstractmethod
    def allocate(
        self, request_id: int, task_pid: int, size_bytes: int, predicted_lifetime_us: int = 0
    ) -> Optional[MemoryHandle]:
        """Attempt to allocate contiguous memory."""
        pass

    @abstractmethod
    def deallocate(self, handle: MemoryHandle) -> bool:
        """Release allocated memory block back to the heap."""
        pass

    @abstractmethod
    def get_metrics(self) -> AllocatorMetrics:
        """Compute current fragmentation and utilization metrics."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset allocator state to initial empty heap."""
        pass
