#!/usr/bin/env python3
"""Check conference_101719.tex for any syntax / delimiter issues."""

import re
from pathlib import Path

tex_file = Path(r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\conference_101719.tex")
lines = tex_file.read_text(encoding="ascii").splitlines()

errors = []
in_verbatim = False
in_equation = False

for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if stripped.startswith("%"):
        continue
    if "\\begin{verbatim}" in line:
        in_verbatim = True
    if "\\end{verbatim}" in line:
        in_verbatim = False
        continue
    if in_verbatim:
        continue
    
    # Check for \mathbf outside math mode or \textbf inside math mode
    text_only = re.sub(r'\$[^$]*\$', '', line)
    if "\\mathbf" in text_only and not "\\begin{equation}" in line and not "\\mathbf{1}" in line:
        errors.append((i, "\\mathbf in text mode", line))
    if "\\mu" in text_only and not "\\documentclass" in text_only and not "\\approx" in line:
        errors.append((i, "\\mu in text mode", line))
    if "\\rho" in text_only:
        errors.append((i, "\\rho in text mode", line))
    if "\\sigma" in text_only:
        errors.append((i, "\\sigma in text mode", line))
    if "\\le" in text_only and not "\\tilde" in line and not "\\clip" in line:
        errors.append((i, "\\le in text mode", line))

    # Count single $
    cleaned_line = line.replace(r"\$", "")
    if "\\begin{equation}" in line:
        in_equation = True
    if "\\end{equation}" in line:
        in_equation = False
        continue
    if not in_equation:
        dollar_count = cleaned_line.count("$")
        if dollar_count % 2 != 0:
            errors.append((i, f"Odd number of $ ({dollar_count})", line))

print(f"Total syntax check errors in conference_101719.tex: {len(errors)}")
for idx, desc, line in errors:
    print(f"  Line {idx}: {desc} -> {line.strip()[:80]}")
