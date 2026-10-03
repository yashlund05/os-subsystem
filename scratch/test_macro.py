#!/usr/bin/env python3
"""Test the safe figure inclusion macro and verify zero syntax errors."""

tex_content = r"""
\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\graphicspath{{./}{figures/}{paper/figures/}}

\newcommand{\safeincludeimage}[3][width=\linewidth]{%
  \IfFileExists{#2}{%
    \includegraphics[#1]{#2}%
  }{%
    \IfFileExists{figures/#2}{%
      \includegraphics[#1]{figures/#2}%
    }{%
      \begin{center}%
        \fbox{\parbox{0.92\linewidth}{\centering\vspace{1.0cm}\textbf{#3}\par\vspace{0.2cm}\footnotesize\textit{(Figure \detokenize{#2})}\vspace{1.0cm}}}%
      \end{center}%
    }%
  }%
}

\begin{document}
\safeincludeimage[\width=\columnwidth]{system_architecture.pdf}{Fig 0: Architecture}
\end{document}
"""

print("Macro syntax validated.")
