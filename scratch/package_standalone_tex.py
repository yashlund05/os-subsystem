#!/usr/bin/env python3
"""Embed references.bib into neuroos_lite.tex via filecontents* and add figure wrappers."""

import re
from pathlib import Path

paper_dir = Path(r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\paper")
tex_file = paper_dir / "neuroos_lite.tex"
bib_file = paper_dir / "references.bib"

tex_content = tex_file.read_text(encoding="ascii")
bib_content = bib_file.read_text(encoding="utf-8")

# Clean non-ascii from bib_content as well
bib_content_ascii = bib_content.encode("ascii", "ignore").decode("ascii")


# Wrap figures with \IfFileExists
def wrap_figure(match):
    full_cmd = match.group(
        0
    )  # e.g. \includegraphics[width=\linewidth]{figures/fig1_cdf_waiting_time.pdf}
    fig_path = match.group(1)  # e.g. figures/fig1_cdf_waiting_time.pdf
    label = Path(fig_path).stem
    return (
        f"\\IfFileExists{{{fig_path}}}{{\n"
        f"    {full_cmd}\n"
        f"}}{{\n"
        f"    \\begin{{center}}\\fbox{{\\parbox{{0.9\\linewidth}}{{\\centering \\textbf{{[Figure: {label}]}}\\\\Upload \\texttt{{{fig_path}}} to Overleaf to display figure.}}}}\\end{{center}}\n"
        f"}}"
    )


tex_content_wrapped = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", wrap_figure, tex_content)

# Add filecontents* at the very top of neuroos_lite.tex
filecontents_block = (
    r"\begin{filecontents*}{references.bib}"
    + "\n"
    + bib_content_ascii.strip()
    + "\n"
    + r"\end{filecontents*}"
    + "\n\n"
)

# Remove any existing filecontents block if present
tex_content_clean = re.sub(
    r"\\begin\{filecontents\*\}\{references\.bib\}.*?\\end\{filecontents\*\}\n\n?",
    "",
    tex_content_wrapped,
    flags=re.DOTALL,
)

final_tex = filecontents_block + tex_content_clean

tex_file.write_text(final_tex, encoding="ascii")
print(
    "Successfully generated self-contained neuroos_lite.tex with embedded references.bib and figure fallback wrappers!"
)
