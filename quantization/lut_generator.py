"""Look-Up Table (LUT) generation for quantum selection and non-linear mappings.

Generates precomputed 256-entry integer tables bounding approximation error to <= 1 LSB,
exporting C header kernel/include/neuroos_lut.h.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class QuantumLUT:
    """Precomputed 256-entry Look-Up Table mapping integer scores to quantum durations."""

    def __init__(
        self,
        q_min_us: int = 1000,
        q_max_us: int = 50000,
        num_entries: int = 256,
        score_min: int = -500000,
        score_max: int = 500000,
    ) -> None:
        self.q_min_us = q_min_us
        self.q_max_us = q_max_us
        self.num_entries = num_entries
        self.score_min = score_min
        self.score_max = score_max

        # Linear ramp across table entries
        # entry 0 -> q_min_us (highest priority / shortest jobs get smaller quantum for responsiveness)
        # entry 255 -> q_max_us (lowest priority / steady background jobs get larger quantum)
        indices = np.arange(num_entries, dtype=np.float64)
        quantums = q_min_us + (indices / (num_entries - 1)) * (q_max_us - q_min_us)
        self.table = np.round(quantums).astype(np.uint32)

    def lookup(self, index: int) -> int:
        idx = max(0, min(self.num_entries - 1, int(index)))
        return int(self.table[idx])

    def score_to_index(self, score: int) -> int:
        clamped = max(self.score_min, min(self.score_max, int(score)))
        frac = (clamped - self.score_min) / max(1, self.score_max - self.score_min)
        idx = int(np.round(frac * (self.num_entries - 1)))
        return max(0, min(self.num_entries - 1, idx))

    def score_to_quantum(self, score: int) -> int:
        idx = self.score_to_index(score)
        return self.lookup(idx)


def generate_c_lut_header(
    lut: QuantumLUT,
    output_path: str = "kernel/include/neuroos_lut.h",
) -> Path:
    """Generates C header exporting the 256-entry lookup table and lookup function."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#ifndef NEUROOS_LUT_H",
        "#define NEUROOS_LUT_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define NEUROOS_LUT_ENTRIES   {lut.num_entries}",
        f"#define NEUROOS_LUT_Q_MIN_US  {lut.q_min_us}",
        f"#define NEUROOS_LUT_Q_MAX_US  {lut.q_max_us}",
        f"#define NEUROOS_LUT_SCORE_MIN {lut.score_min}",
        f"#define NEUROOS_LUT_SCORE_MAX {lut.score_max}",
        "",
        "/* Precomputed 256-entry LUT: error bounded to <= 1 LSB */",
        f"static const uint32_t neuroos_quantum_lut[{lut.num_entries}] = {{",
    ]

    # Format 8 values per line
    for row_start in range(0, lut.num_entries, 8):
        chunk = [f"{val}U" for val in lut.table[row_start : row_start + 8]]
        comma = "," if row_start + 8 < lut.num_entries else ""
        lines.append("    " + ", ".join(chunk) + comma)

    lines.extend(
        [
            "};",
            "",
            "static inline uint32_t neuroos_lut_lookup(int32_t score) {",
            "    if (score <= NEUROOS_LUT_SCORE_MIN) return neuroos_quantum_lut[0];",
            "    if (score >= NEUROOS_LUT_SCORE_MAX) return neuroos_quantum_lut[NEUROOS_LUT_ENTRIES - 1];",
            "    int64_t span = (int64_t)NEUROOS_LUT_SCORE_MAX - (int64_t)NEUROOS_LUT_SCORE_MIN;",
            "    int64_t offset = (int64_t)score - (int64_t)NEUROOS_LUT_SCORE_MIN;",
            "    uint32_t idx = (uint32_t)((offset * (NEUROOS_LUT_ENTRIES - 1)) / span);",
            "    if (idx >= NEUROOS_LUT_ENTRIES) idx = NEUROOS_LUT_ENTRIES - 1;",
            "    return neuroos_quantum_lut[idx];",
            "}",
            "",
            "#endif /* NEUROOS_LUT_H */",
            "",
        ]
    )

    content = "\n".join(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    return out
