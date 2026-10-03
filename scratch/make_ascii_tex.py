#!/usr/bin/env python3
"""Completely convert neuroos_lite.tex to 100% ASCII LaTeX."""

import re
from pathlib import Path

tex_path = Path(r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\paper\neuroos_lite.tex")

content = tex_path.read_text(encoding="utf-8")

# Structural / unicode map
replacements = {
    "─": "-",
    "—": "---",
    "–": "--",
    "’": "'",
    "‘": "`",
    "“": "``",
    "”": "''",
    "−": "-",
    "…": r"\dots ",
    "×": r"\times ",
    "±": r"\pm ",
    "ρ": r"\rho ",
    "σ": r"\sigma ",
    "μ": r"\mu ",
    "α": r"\alpha ",
    "γ": r"\gamma ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "≈": r"\approx ",
    "€": "EUR",
    "\u00a0": " ",
}

for k, v in replacements.items():
    content = content.replace(k, v)

# Check for any residual non-ascii chars
residual = []
for i, char in enumerate(content):
    if ord(char) > 127:
        residual.append((ord(char), hex(ord(char)), char))

if residual:
    print(f"Residual non-ASCII count: {len(residual)}")
    unique_residual = set(residual)
    for code, hexcode, ch in unique_residual:
        print(f"  Code: {code} ({hexcode})")
        # remove residual non-ascii
        content = content.replace(ch, "")

tex_path.write_text(content, encoding="ascii", errors="strict")
print("Successfully converted neuroos_lite.tex to 100% 7-bit ASCII!")
