"""Python-layer SPSC ring buffer (mirrors telemetry/ring_buffer/ring_buffer.c).

Used by:
- tests/integration/test_integration_pipeline.py
- userspace/telemetry/daemon.py (drain loop)

This is a pure-Python simulation of the C lock-free SPSC ring buffer.
The C version uses atomic operations; this version uses a simple deque
with the same API contract: push/pop return bool, capacity is fixed.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional


class RingBuffer:
    """Pure-Python SPSC ring buffer with capacity-bounded FIFO semantics.

    Mirrors the API of the C ``spsc_ring_buffer`` defined in
    ``telemetry/ring_buffer/include/ring_buffer.h``.
    """

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        # Round up to next power of two (mirrors C implementation)
        cap = 1
        while cap < capacity:
            cap <<= 1
        self._capacity: int = cap
        self._buf: Deque[Any] = deque()

    # ------------------------------------------------------------------
    # Core API (mirrors C functions)
    # ------------------------------------------------------------------

    def push(self, item: Any) -> bool:
        """Push an item. Returns False (without blocking) if the buffer is full."""
        if len(self._buf) >= self._capacity:
            return False
        self._buf.append(item)
        return True

    def pop(self) -> Optional[Any]:
        """Pop the oldest item. Returns None if the buffer is empty."""
        if not self._buf:
            return None
        return self._buf.popleft()

    def empty(self) -> bool:
        """Return True if no items are available."""
        return len(self._buf) == 0

    def full(self) -> bool:
        """Return True if the buffer has reached its capacity."""
        return len(self._buf) >= self._capacity

    def count(self) -> int:
        """Return the number of items currently in the buffer."""
        return len(self._buf)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, count={len(self._buf)})"
