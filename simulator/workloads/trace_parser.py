"""Cluster trace parser for Google Borg and SPEC CPU2017 workload replay."""

import csv
import io
from typing import List

from simulator.scheduling.task import SimulatedTask


class ClusterTraceParser:
    """
    Parses uniprocessor cluster trace slices (Google Borg, SPEC CPU2017 profiles).
    Supports CSV and tab-delimited streams with columns:
    arrival_us, pid, burst_us, [cache_misses, branch_mispred, mem_kb]
    """

    @staticmethod
    def parse_csv_stream(stream: io.TextIOBase) -> List[SimulatedTask]:
        tasks: List[SimulatedTask] = []
        reader = csv.DictReader(stream)

        for row in reader:
            pid = int(row.get("pid", row.get("task_id", 0)))
            arrival = int(float(row.get("arrival_us", row.get("start_time", 0))))
            burst = int(float(row.get("burst_us", row.get("duration", 1000))))
            cache_misses = int(float(row.get("cache_misses", 0)))
            branch_mispred = int(float(row.get("branch_mispred", 0)))
            mem_kb = int(float(row.get("mem_kb", row.get("memory_kb", 4096))))

            tasks.append(
                SimulatedTask(
                    pid=pid,
                    arrival_time_us=arrival,
                    total_burst_us=max(1, burst),
                    cache_misses=cache_misses,
                    branch_mispredictions=branch_mispred,
                    memory_footprint_kb=mem_kb,
                )
            )

        return sorted(tasks, key=lambda t: (t.arrival_time_us, t.pid))

    @staticmethod
    def parse_csv_file(file_path: str) -> List[SimulatedTask]:
        with open(file_path, "r", encoding="utf-8") as f:
            return ClusterTraceParser.parse_csv_stream(f)
