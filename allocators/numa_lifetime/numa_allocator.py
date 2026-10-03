"""NUMA-Aware Lifetime Affinity Memory Allocator (Phase 6).

Extends the single-domain lifetime-affinity clustering allocator:
- Partitions system memory across K NUMA nodes.
- Each NUMA node maintains independent lifetime affinity bands (<5ms, <20ms, <100ms, <500ms, >=500ms).
- Primary allocation targets the local NUMA node of the requesting CPU thread.
- If local band is exhausted, probes neighbor bands on the same NUMA node first before remote nodes.
- Preserves interconnect bandwidth (QPI/UPI) while maintaining anti-fragmentation guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from allocators.base import AllocatorMetrics, BaseAllocator, MemoryHandle
from allocators.buddy.allocator import BuddyAllocator

BAND_EDGES_US = [5000, 20000, 100000, 500000]


def lifetime_to_band(tau_us: int) -> int:
    for i, edge in enumerate(BAND_EDGES_US):
        if tau_us < edge:
            return i
    return len(BAND_EDGES_US)


@dataclass
class NUMABandBlock:
    offset: int
    size_bytes: int
    node_id: int
    is_allocated: bool = False
    handle: Optional[MemoryHandle] = None


class NUMALifetimeAllocator(BaseAllocator):
    """NUMA-partitioned lifetime affinity allocator."""

    def __init__(
        self,
        total_heap_bytes: int = 64 * 1024 * 1024,
        num_nodes: int = 2,
        min_split_bytes: int = 64,
    ) -> None:
        super().__init__(name="NUMA-NeuroOS-Mem", total_heap_bytes=total_heap_bytes)
        self.num_nodes = max(1, num_nodes)
        self.min_split_bytes = min_split_bytes
        self.num_bands = len(BAND_EDGES_US) + 1

        # Partition total heap equally across NUMA nodes
        self.node_heap_bytes = total_heap_bytes // self.num_nodes

        # node_id -> band_idx -> list of NUMABandBlock
        self.node_bands: List[List[List[NUMABandBlock]]] = []
        band_size = self.node_heap_bytes // self.num_bands

        for n in range(self.num_nodes):
            node_bands = []
            node_start = n * self.node_heap_bytes
            for b in range(self.num_bands):
                start = node_start + b * band_size
                size = (
                    band_size
                    if b < self.num_bands - 1
                    else (self.node_heap_bytes - (b * band_size))
                )
                node_bands.append([NUMABandBlock(offset=start, size_bytes=size, node_id=n)])
            self.node_bands.append(node_bands)

        self._handles: Dict[int, Tuple[int, int, NUMABandBlock]] = {}
        self._next_id = 1

        # Fallback global buddy allocator
        self.fallback = BuddyAllocator(total_heap_bytes=total_heap_bytes)
        self.fallback_trips = 0
        self.local_node_allocations = 0
        self.remote_node_allocations = 0

    def _band_alloc(
        self, node_id: int, band_idx: int, size_bytes: int, task_pid: int, tau_us: int
    ) -> Optional[MemoryHandle]:
        band = self.node_bands[node_id][band_idx]
        for idx, blk in enumerate(band):
            if not blk.is_allocated and blk.size_bytes >= size_bytes:
                excess = blk.size_bytes - size_bytes
                if excess >= self.min_split_bytes:
                    new_blk = NUMABandBlock(
                        offset=blk.offset + size_bytes,
                        size_bytes=excess,
                        node_id=node_id,
                    )
                    blk.size_bytes = size_bytes
                    band.insert(idx + 1, new_blk)
                blk.is_allocated = True
                handle = MemoryHandle(
                    block_id=self._next_id,
                    offset=blk.offset,
                    size_bytes=blk.size_bytes,
                    requested_size_bytes=size_bytes,
                    task_pid=task_pid,
                    predicted_lifetime_us=tau_us,
                )
                self._next_id += 1
                blk.handle = handle
                self._handles[handle.block_id] = (node_id, band_idx, blk)
                return handle
        return None

    def allocate(
        self,
        request_id: int,
        task_pid: int,
        size_bytes: int,
        predicted_lifetime_us: int = 0,
        preferred_node: int = 0,
    ) -> Optional[MemoryHandle]:
        self.total_alloc_requests += 1
        if size_bytes <= 0 or size_bytes > self.total_heap_bytes:
            self.failed_alloc_requests += 1
            return None

        preferred_node = preferred_node % self.num_nodes
        band_idx = lifetime_to_band(predicted_lifetime_us)

        # 1. Primary allocation: local NUMA node designated lifetime band
        handle = self._band_alloc(
            preferred_node, band_idx, size_bytes, task_pid, predicted_lifetime_us
        )
        if handle is not None:
            self.local_node_allocations += 1
            return handle

        # 2. Probe adjacent bands on the SAME local NUMA node
        for delta in (1, -1, 2, -2):
            nb = band_idx + delta
            if 0 <= nb < self.num_bands:
                handle = self._band_alloc(
                    preferred_node, nb, size_bytes, task_pid, predicted_lifetime_us
                )
                if handle is not None:
                    self.local_node_allocations += 1
                    return handle

        # 3. Probe remote NUMA nodes before falling back to Buddy
        for n in range(self.num_nodes):
            if n == preferred_node:
                continue
            handle = self._band_alloc(n, band_idx, size_bytes, task_pid, predicted_lifetime_us)
            if handle is not None:
                self.remote_node_allocations += 1
                return handle

        # 4. Buddy fallback
        self.fallback_trips += 1
        return self.fallback.allocate(request_id, task_pid, size_bytes, predicted_lifetime_us)

    def deallocate(self, handle: MemoryHandle) -> bool:
        if handle.block_id in self._handles:
            node_id, band_idx, blk = self._handles.pop(handle.block_id)
            blk.is_allocated = False
            blk.handle = None
            self._coalesce_band(node_id, band_idx)
            return True
        return self.fallback.deallocate(handle)

    def _coalesce_band(self, node_id: int, band_idx: int) -> None:
        band = self.node_bands[node_id][band_idx]
        i = 0
        while i < len(band) - 1:
            curr = band[i]
            nxt = band[i + 1]
            if (
                (not curr.is_allocated)
                and (not nxt.is_allocated)
                and (curr.offset + curr.size_bytes == nxt.offset)
            ):
                curr.size_bytes += nxt.size_bytes
                band.pop(i + 1)
            else:
                i += 1

    def get_metrics(self) -> AllocatorMetrics:
        free_bytes = 0
        largest_free = 0
        allocated_bytes = 0
        requested_bytes = 0

        for node_bands in self.node_bands:
            for band in node_bands:
                for blk in band:
                    if blk.is_allocated:
                        allocated_bytes += blk.size_bytes
                        if blk.handle:
                            requested_bytes += blk.handle.requested_size_bytes
                    else:
                        free_bytes += blk.size_bytes
                        if blk.size_bytes > largest_free:
                            largest_free = blk.size_bytes

        ext_frag = 1.0 - (float(largest_free) / float(free_bytes)) if free_bytes > 0 else 0.0
        int_frag = (
            max(0.0, (allocated_bytes - requested_bytes) / allocated_bytes)
            if allocated_bytes > 0
            else 0.0
        )
        buf_util = requested_bytes / self.total_heap_bytes if self.total_heap_bytes else 0.0

        return AllocatorMetrics(
            total_heap_bytes=self.total_heap_bytes,
            allocated_bytes=allocated_bytes,
            free_bytes=free_bytes,
            max_free_block_bytes=largest_free,
            external_fragmentation=max(0.0, ext_frag),
            internal_fragmentation=int_frag,
            buffer_utilization=buf_util,
            total_alloc_requests=self.total_alloc_requests,
            failed_alloc_requests=self.failed_alloc_requests,
            total_dealloc_requests=self.total_dealloc_requests,
        )

    def compute_metrics(self) -> AllocatorMetrics:
        return self.get_metrics()

    def reset(self) -> None:
        self.node_bands = []
        band_size = self.node_heap_bytes // self.num_bands
        for n in range(self.num_nodes):
            node_bands = []
            node_start = n * self.node_heap_bytes
            for b in range(self.num_bands):
                start = node_start + b * band_size
                size = (
                    band_size
                    if b < self.num_bands - 1
                    else (self.node_heap_bytes - (b * band_size))
                )
                node_bands.append([NUMABandBlock(offset=start, size_bytes=size, node_id=n)])
            self.node_bands.append(node_bands)
        self._handles.clear()
        self.fallback.reset()
        self._next_id = 1
        self.total_alloc_requests = 0
        self.failed_alloc_requests = 0
        self.total_dealloc_requests = 0
        self.fallback_trips = 0
        self.local_node_allocations = 0
        self.remote_node_allocations = 0
