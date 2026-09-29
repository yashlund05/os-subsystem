"""Lifetime-affinity clustering allocator (Phase 3 Week 6).

Per docs/TRD.md section 5 + docs/Architecture.md section 4:
- Request R_k = <S_k, tau_k>; colocate correlated deallocation horizons into
  contiguous bands so blocks free together, suppressing external fragmentation.
- Bands: 4 lifetime classes by tau_k edges [5ms, 20ms, 100ms, 500ms].
- Within-band first-fit with split/coalesce; bounded O(1)/O(log N) lookup.
- Guardrail: if band cannot satisfy within epsilon budget, fallback to Buddy.

Simulator implementation (Python mirror of kernel slab interceptor).
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
class BandBlock:
    offset: int
    size_bytes: int
    is_allocated: bool = False
    handle: Optional[MemoryHandle] = None


class LifetimeAffinityAllocator(BaseAllocator):
    """Banded lifetime-affinity allocator with Buddy fallback."""

    def __init__(
        self, total_heap_bytes: int = 64 * 1024 * 1024, min_split_bytes: int = 64
    ) -> None:
        super().__init__(name="NeuroOS-Lite-Mem", total_heap_bytes=total_heap_bytes)
        self.min_split_bytes = min_split_bytes
        self.num_bands = len(BAND_EDGES_US) + 1
        # Partition heap into equal bands by address (contiguous zones)
        band_size = total_heap_bytes // self.num_bands
        self.bands: List[List[BandBlock]] = []
        for b in range(self.num_bands):
            start = b * band_size
            size = band_size if b < self.num_bands - 1 else (total_heap_bytes - start)
            self.bands.append([BandBlock(offset=start, size_bytes=size)])
        self._handles: Dict[int, tuple[int, BandBlock]] = {}
        self._next_id = 1
        self.fallback = BuddyAllocator(total_heap_bytes=total_heap_bytes)
        self.fallback_trips = 0

    def _band_alloc(
        self, band_idx: int, size_bytes: int, task_pid: int, tau_us: int
    ) -> Optional[MemoryHandle]:
        band = self.bands[band_idx]
        for idx, blk in enumerate(band):
            if not blk.is_allocated and blk.size_bytes >= size_bytes:
                excess = blk.size_bytes - size_bytes
                if excess >= self.min_split_bytes:
                    new_blk = BandBlock(offset=blk.offset + size_bytes, size_bytes=excess)
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
                self._handles[handle.block_id] = (band_idx, blk)
                return handle
        return None

    def allocate(
        self, request_id: int, task_pid: int, size_bytes: int, predicted_lifetime_us: int = 0
    ) -> Optional[MemoryHandle]:
        self.total_alloc_requests += 1
        if size_bytes <= 0 or size_bytes > self.total_heap_bytes:
            self.failed_alloc_requests += 1
            return None
        band_idx = lifetime_to_band(predicted_lifetime_us)
        handle = self._band_alloc(band_idx, size_bytes, task_pid, predicted_lifetime_us)
        if handle is not None:
            return handle
        # Bounded neighbor-band probe (epsilon budget: 2 neighbor bands, O(1))
        for delta in (1, -1, 2, -2):
            nb = band_idx + delta
            if 0 <= nb < self.num_bands:
                handle = self._band_alloc(nb, size_bytes, task_pid, predicted_lifetime_us)
                if handle is not None:
                    return handle
        # Guardrail fallback to Buddy (TRD section 6)
        self.fallback_trips += 1
        fb = self.fallback.allocate(request_id, task_pid, size_bytes, predicted_lifetime_us)
        if fb is None:
            self.failed_alloc_requests += 1
            return None
        # Re-tag fallback handle into our id space for unified deallocate
        handle = MemoryHandle(
            block_id=self._next_id,
            offset=fb.offset,
            size_bytes=fb.size_bytes,
            requested_size_bytes=fb.requested_size_bytes,
            task_pid=task_pid,
            predicted_lifetime_us=predicted_lifetime_us,
        )
        self._next_id += 1
        # Track as fallback-backed (band_idx = -1)
        marker = BandBlock(offset=handle.offset, size_bytes=handle.size_bytes, is_allocated=True)
        marker.handle = handle
        self._handles[handle.block_id] = (-1, marker)
        # Store buddy handle for later free (keyed by our id)
        self._handles[handle.block_id] = (-1, marker)
        # Keep buddy allocation alive via extra map
        if not hasattr(self, "_fb_map"):
            self._fb_map: Dict[int, MemoryHandle] = {}
        self._fb_map[handle.block_id] = fb
        return handle

    def deallocate(self, handle: MemoryHandle) -> bool:
        self.total_dealloc_requests += 1
        entry = self._handles.pop(handle.block_id, None)
        if entry is None:
            return False
        band_idx, blk = entry
        if band_idx == -1:
            fb = getattr(self, "_fb_map", {}).pop(handle.block_id, None)
            if fb is not None:
                return bool(self.fallback.deallocate(fb))
            return True
        blk.is_allocated = False
        blk.handle = None
        self._coalesce_band(band_idx)
        return True

    def _coalesce_band(self, band_idx: int) -> None:
        band = self.bands[band_idx]
        i = 0
        while i < len(band) - 1:
            a, b = band[i], band[i + 1]
            if not a.is_allocated and not b.is_allocated:
                a.size_bytes += b.size_bytes
                band.pop(i + 1)
            else:
                i += 1

    def get_metrics(self) -> AllocatorMetrics:
        free_sizes: List[int] = []
        alloc_bytes = 0
        req_bytes = 0
        for band in self.bands:
            for blk in band:
                if blk.is_allocated and blk.handle is not None:
                    alloc_bytes += blk.size_bytes
                    req_bytes += blk.handle.requested_size_bytes
                else:
                    free_sizes.append(blk.size_bytes)
        # Include fallback Buddy occupancy
        fb_m = self.fallback.get_metrics()
        alloc_bytes += fb_m.allocated_bytes
        free_bytes = sum(free_sizes) + fb_m.free_bytes if free_sizes or fb_m.free_bytes else 0
        # Recompute free from heap accounting to stay consistent
        free_bytes = max(0, self.total_heap_bytes - alloc_bytes)
        max_free = max(free_sizes) if free_sizes else 0
        # Only consider Buddy fallback max when fallback actually backs allocations;
        # otherwise its untouched full-heap free list would mask real band fragmentation.
        if fb_m.allocated_bytes > 0:
            max_free = max(max_free, fb_m.max_free_block_bytes)
        # Cap by actual free bytes to keep Frag_ext in [0,1].
        max_free = min(max_free, free_bytes)
        ext = 1.0 - (max_free / free_bytes) if free_bytes > 0 else 0.0
        int_frag = max(0.0, (alloc_bytes - req_bytes) / alloc_bytes) if alloc_bytes > 0 else 0.0
        buf_util = req_bytes / self.total_heap_bytes if self.total_heap_bytes else 0.0
        return AllocatorMetrics(
            total_heap_bytes=self.total_heap_bytes,
            allocated_bytes=alloc_bytes,
            free_bytes=free_bytes,
            max_free_block_bytes=max_free,
            external_fragmentation=ext,
            internal_fragmentation=int_frag,
            buffer_utilization=buf_util,
            total_alloc_requests=self.total_alloc_requests,
            failed_alloc_requests=self.failed_alloc_requests,
            total_dealloc_requests=self.total_dealloc_requests,
        )

    def reset(self) -> None:
        band_size = self.total_heap_bytes // self.num_bands
        self.bands = []
        for b in range(self.num_bands):
            start = b * band_size
            size = band_size if b < self.num_bands - 1 else (self.total_heap_bytes - start)
            self.bands.append([BandBlock(offset=start, size_bytes=size)])
        self._handles.clear()
        if hasattr(self, "_fb_map"):
            self._fb_map.clear()
        self.fallback.reset()
        self._next_id = 1
        self.total_alloc_requests = 0
        self.failed_alloc_requests = 0
        self.total_dealloc_requests = 0
        self.fallback_trips = 0
