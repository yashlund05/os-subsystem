#!/usr/bin/env python3
"""Check and clean remaining \\SI or \\si calls in paper/neuroos_lite.tex."""

import re
from pathlib import Path

tex_path = Path(
    r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\paper\neuroos_lite.tex"
)
content = tex_path.read_text(encoding="ascii")

# Find all \SI or \si calls
si_matches = re.findall(r"\\S[Ii]\{[^}]*\}*(?:\{[^}]*\})*", content)
print(f"Total \\SI / \\si instances found: {len(si_matches)}")
for match in set(si_matches[:30]):
    print("  ", match)

# Let's replace any remaining \SI{val}{unit} with clean math mode expressions
# e.g., \SI{50}{\nano\second} -> 50\,ns
# \SI{702.8}{\microsecond} -> 702.8\,\mu s
# \SI{1.5}{\microsecond} -> 1.5\,\mu s
# \si{\microsecond} -> \mu s


def fix_si(m):
    text = m.group(0)
    # \SI{val}{unit}
    m2 = re.match(r"\\SI\{([^}]+)\}(?:\{([^}]+)\})?", text)
    if not m2:
        return text
    val = m2.group(1)
    unit = m2.group(2) if m2.group(2) else ""

    # clean unit
    unit = unit.replace(r"\microsecond", r"\mu\text{s}")
    unit = unit.replace(r"\micro\second", r"\mu\text{s}")
    unit = unit.replace(r"\nano\second", r"\text{ns}")
    unit = unit.replace(r"\milli\second", r"\text{ms}")
    unit = unit.replace(r"\giga\hertz", r"\text{GHz}")
    unit = unit.replace(r"\cycle", r"\text{cycles}")
    unit = unit.replace(r"\us", r"\mu\text{s}")
    unit = unit.replace(r"\ns", r"\text{ns}")
    unit = unit.replace(r"\ms", r"\text{ms}")

    if unit:
        return f"{val}\\,{unit}"
    else:
        return val


cleaned_content = re.sub(r"\\SI\{[^}]+\}(?:\{[^}]*\})?", fix_si, content)
cleaned_content = cleaned_content.replace(r"\si{\microsecond}", r"\mu\text{s}")
cleaned_content = cleaned_content.replace(r"\si{\micro\second}", r"\mu\text{s}")
cleaned_content = cleaned_content.replace(r"\usepackage{siunitx}", "")

tex_path.write_text(cleaned_content, encoding="ascii")
print("Cleaned all \\SI and \\si macros successfully!")
