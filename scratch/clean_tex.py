#!/usr/bin/env python3
"""Clean LaTeX file of non-ASCII UTF-8 characters and fragile macro combinations."""

import re
from pathlib import Path

tex_path = Path(
    r"c:\Users\Aditya\Downloads\os-subsystem-master\os-subsystem-master\paper\neuroos_lite.tex"
)

content = tex_path.read_text(encoding="utf-8")

# Replacements map for unicode characters
unicode_replacements = {
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
    "™": "",
    "®": "",
}

for orig, repl in unicode_replacements.items():
    content = content.replace(orig, repl)

# Replace problematic siunitx macros with robust standard LaTeX math formatting
# e.g. \SI{0.8}--\SI{2.5}{\microsecond} -> 0.8--2.5\,\mu\text{s}
content = re.sub(
    r"\\SI\{([0-9\.]+)\}\s*--\s*\\SI\{([0-9\.]+)\}\{\\microsecond\}",
    r"\1--\2\\,\\mu\\text{s}",
    content,
)
content = re.sub(
    r"\\SI\{([0-9\.]+)\}\s*--\s*\\SI\{([0-9\.]+)\}\{\\micro\s*second\}",
    r"\1--\2\\,\\mu\\text{s}",
    content,
)
content = re.sub(r"\\SI\{([0-9\.]+)\}\{\\microsecond\}", r"\1\\,\\mu\\text{s}", content)
content = re.sub(r"\\SI\{([0-9\.]+)\}\{\\nano\\second\}", r"\1\\,\\text{ns}", content)
content = re.sub(r"\\SI\{([0-9\.]+)\}\{\\milli\\second\}", r"\1\\,\\text{ms}", content)
content = re.sub(r"\\SI\{([0-9\.]+)\}\{\\giga\\hertz\}", r"\1\\,\\text{GHz}", content)
content = re.sub(r"\\SI\{([0-9\.]+)\}\{\\cycle\}", r"\1\\,\\text{cycles}", content)
content = re.sub(r"\\SI\{<([0-9\.]+)\}\{\\cycle\}", r"<\1\\,\\text{cycles}", content)
content = re.sub(r"\\SI\{<([0-9\.]+)\}\{\\nano\\second\}", r"<\1\\,\\text{ns}", content)
content = re.sub(r"\\SI\{([0-9,]+)\}\{\\microsecond\}", r"\1\\,\\mu\\text{s}", content)
content = re.sub(r"\\SI\{([0-9,]+)\}\{\\milli\\second\}", r"\1\\,\\text{ms}", content)

content = content.replace(r"\si{\microsecond}", r"\mu\text{s}")
content = content.replace(r"\DeclareSIUnit{\cycle}{cycle}", "")
content = content.replace(r"\DeclareSIUnit{\microsecond}{\micro s}", "")

tex_path.write_text(content, encoding="utf-8")
print("Cleaned neuroos_lite.tex successfully.")
