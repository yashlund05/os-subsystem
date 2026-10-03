#!/usr/bin/env python3
"""Check paper/neuroos_lite.tex for non-ASCII characters or macro issues."""

from pathlib import Path

tex_path = Path(r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\paper\neuroos_lite.tex")
lines = tex_path.read_text(encoding="utf-8").splitlines()

non_ascii_found = []
for i, line in enumerate(lines, 1):
    for ch in line:
        if ord(ch) > 127:
            non_ascii_found.append((i, ch, repr(ch), line))
            break

print(f"Total non-ASCII lines found: {len(non_ascii_found)}")
for idx, ch, repr_ch, line in non_ascii_found[:15]:
    print(f"Line {idx}: {repr_ch} -> {line.strip()[:80]}")
