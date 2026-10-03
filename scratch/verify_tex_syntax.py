#!/usr/bin/env python3
"""Check math mode delimiters and syntax errors in paper/neuroos_lite.tex."""

import re
from pathlib import Path

tex_file = Path(r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\paper\neuroos_lite.tex")
lines = tex_file.read_text(encoding="ascii").splitlines()

# 1. Check $ balance per line (ignoring escaped \$)
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
    # Remove math mode blocks to inspect text mode
    text_only = re.sub(r'\$[^$]*\$', '', line)
    if "\\mathbf" in text_only:
        errors.append((i, "\\mathbf in text mode", line))
    if "\\mu" in text_only and not "\\documentclass" in text_only:
        errors.append((i, "\\mu in text mode", line))
    if "\\rho" in text_only:
        errors.append((i, "\\rho in text mode", line))
    if "\\sigma" in text_only:
        errors.append((i, "\\sigma in text mode", line))
    if "\\alpha" in text_only:
        errors.append((i, "\\alpha in text mode", line))
    if "\\le" in text_only:
        errors.append((i, "\\le in text mode", line))
    if "\\ge" in text_only:
        errors.append((i, "\\ge in text mode", line))
    if "\\to" in text_only and not "\\mbox" in text_only:
        errors.append((i, "\\to in text mode", line))

    # Count single $ (ignoring \$)
    cleaned_line = line.replace(r"\$", "")
    # ignore lines inside equation environment
    if "\\begin{equation}" in line:
        in_equation = True
    if "\\end{equation}" in line:
        in_equation = False
        continue
    if not in_equation:
        dollar_count = cleaned_line.count("$")
        if dollar_count % 2 != 0:
            errors.append((i, f"Odd number of $ ({dollar_count})", line))

print(f"Total syntax check errors found: {len(errors)}")
for idx, desc, line in errors:
    print(f"  Line {idx}: {desc} -> {line.strip()[:80]}")
