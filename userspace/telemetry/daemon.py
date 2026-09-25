"""User-Space Telemetry Ingestion Daemon."""

import ctypes
import mmap
import os
import time
from collections import defaultdict
from typing import List


# Definition of the 16-byte packed task_telemetry struct
class TaskTelemetry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("pid", ctypes.c_uint32),
        ("elapsed_us", ctypes.c_uint16),
        ("cache_misses_delta", ctypes.c_uint16),
        ("branch_mispred_delta", ctypes.c_uint16),
        ("mem_footprint_kb", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("_pad", ctypes.c_uint16),
    ]


class SPSCRingBuffer(ctypes.Structure):
    _fields_ = [
        ("capacity", ctypes.c_uint32),
        ("mask", ctypes.c_uint32),
        ("head", ctypes.c_uint32),
        ("pad1", ctypes.c_uint8 * 60),
        ("tail", ctypes.c_uint32),
        ("pad2", ctypes.c_uint8 * 60),
        # Flexible array member for entries is handled separately
    ]


class TelemetryDaemon:
    def __init__(self, shm_path: str = "/dev/shm/neuroos_telemetry"):
        self.shm_path = shm_path
        self.running = False
        self.stats = defaultdict(
            lambda: {
                "total_burst_us": 0,
                "cache_misses": 0,
                "branch_mispredictions": 0,
                "samples": 0,
            }
        )

    def process_batch(self, batch: List[TaskTelemetry]):
        """Compute rolling statistics (IPC, mean cache miss deltas, burst variances)."""
        for item in batch:
            pid = item.pid
            self.stats[pid]["total_burst_us"] += item.elapsed_us
            self.stats[pid]["cache_misses"] += item.cache_misses_delta
            self.stats[pid]["branch_mispredictions"] += item.branch_mispred_delta
            self.stats[pid]["samples"] += 1

            # Additional rolling stat logic for feature pipelines would go here
            # e.g. EWMA for burst variances

    def run(self):
        """Poll lock-free SPSC ring buffer without introducing lock contention."""
        if not os.path.exists(self.shm_path):
            print(f"Warning: {self.shm_path} does not exist. Waiting...")

        while not os.path.exists(self.shm_path) and self.running:
            time.sleep(1.0)

        if not self.running:
            return

        with open(self.shm_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)

            # Point to the header
            rb_header = SPSCRingBuffer.from_buffer(mm)

            entries_offset = ctypes.sizeof(SPSCRingBuffer)
            entry_size = ctypes.sizeof(TaskTelemetry)
            mask = rb_header.mask

            while self.running:
                head = rb_header.head
                tail = rb_header.tail

                if head == tail:
                    # Empty, backoff
                    time.sleep(0.001)
                    continue

                batch = []
                # Read up to head
                while tail != head:
                    index = tail & mask
                    offset = entries_offset + index * entry_size

                    entry = TaskTelemetry.from_buffer(mm, offset)

                    # Need to make a copy since the buffer will be overwritten
                    copy_entry = TaskTelemetry()
                    ctypes.memmove(
                        ctypes.addressof(copy_entry), ctypes.addressof(entry), entry_size
                    )
                    batch.append(copy_entry)

                    tail += 1

                # Process the batch
                self.process_batch(batch)

                # Update tail to consume
                rb_header.tail = tail

    def start(self):
        self.running = True
        self.run()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    daemon = TelemetryDaemon()
    try:
        daemon.start()
    except KeyboardInterrupt:
        daemon.stop()
