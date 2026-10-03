#!/usr/bin/env python3
"""Check all underscores and LaTeX syntax in conference_101719.tex."""

import re
from pathlib import Path

tex_file = Path(r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\paper\conference_101719.tex")
lines = tex_file.read_text(encoding="ascii").splitlines()

# Search for unescaped _ outside of math mode, verbatim, and bibtex
in_verbatim = False
in_filecontents = False

for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped.startswith("%"):
        continue
    if "\\begin{filecontents*}" in line:
        in_filecontents = True
    if "\\end{filecontents*}" in line:
        in_filecontents = False
        continue
    if in_filecontents:
        continue
    if "\\begin{verbatim}" in line:
        in_verbatim = True
    if "\\end{verbatim}" in line:
        in_verbatim = False
        continue
    if in_verbatim:
        continue
    
    # Remove $...$ math mode
    no_math = re.sub(r'\$[^$]*\$', '', line)
    # Remove equation environments
    # Remove \_
    no_escaped = no_math.replace(r"\_", "")
    if "_" in no_escaped and not "\\begin{equation}" in line:
        print(f"Line {i}: unescaped underscore -> {line}")
