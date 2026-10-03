# NeuroOS-Lite IEEE Paper Build Makefile
# Usage:
#   make         — Full build (pdflatex + bibtex cycle)
#   make figures — Regenerate publication figures from data
#   make clean   — Remove build artefacts
#   make open    — Open PDF (Linux: xdg-open, macOS: open)

PAPER   := neuroos_lite
LATEX   := pdflatex
BIBTEX  := bibtex
PYTHON  := python3
FIGSCRIPT := ../scripts/reproduce/generate_figures.py

.PHONY: all figures clean open check

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib figures
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	@echo "[✓] PDF built: $(PAPER).pdf"

figures:
	@echo "Generating publication figures..."
	cd .. && $(PYTHON) $(FIGSCRIPT)
	@echo "[✓] Figures generated in paper/figures/"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk
	@echo "[✓] Build artefacts removed"

open: $(PAPER).pdf
	@if command -v xdg-open > /dev/null 2>&1; then \
	    xdg-open $(PAPER).pdf; \
	elif command -v open > /dev/null 2>&1; then \
	    open $(PAPER).pdf; \
	else \
	    echo "Please open $(PAPER).pdf manually."; \
	fi

check:
	@echo "Checking LaTeX installation..."
	@$(LATEX) --version | head -1
	@$(BIBTEX) --version | head -1
	@$(PYTHON) --version
