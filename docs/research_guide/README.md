# From Intent to Silicon — research guide

A step-by-step guide for an independent research project that takes this
thesis into VLSI-based DSP, ML hardware acceleration and FPGA/ASIC
implementation.

**Read the PDF:** [`research_guide.pdf`](research_guide.pdf)

## What is inside

| Part | Chapters |
|---|---|
| I — Foundations | Project overview and research questions · hardware primer · the thesis as a hardware workload (measured) |
| II — The project, step by step | Phase 0 setup · Phase 1 co-design and fixed-point golden model · Phase 2 RTL · Phase 3 verification · Phase 4 FPGA prototype (PYNQ-Z2) · Phase 5A ML acceleration · Phase 5B VLSI/ASIC |
| III — Doing it well | Evaluation methodology · writing up and presenting · timeline, milestones and risks |
| Appendices | Measured fact sheet · register map · command cheat sheet · glossary · templates · reading list |

## Building

Requires a TeX distribution with `pdflatex`, `biber` and `latexmk` (TeX Live
2023 or newer; on Debian/Ubuntu: `texlive-latex-extra texlive-fonts-extra
texlive-bibtex-extra texlive-science texlive-pictures biber latexmk`).

```bash
cd docs/research_guide
make            # builds build/main.pdf and copies it to research_guide.pdf
make clean
```

To check a single chapter quickly in an isolated build directory:

```bash
tools/check_chapter.sh chapters/05-codesign
```

## Reproducing the numbers

Every number about the thesis system in the guide is either measured by
[`scripts/measure_workload.py`](scripts/measure_workload.py) or marked as an
estimate. From the repository root:

```bash
PYTHONPATH=src python docs/research_guide/scripts/measure_workload.py
```

## Layout

```
main.tex            document skeleton (parts, chapters, appendices)
preamble.tex        packages, colours, step/tip/pitfall boxes, macros
frontmatter/        title page, how to use this guide
chapters/           one file per chapter
appendices/         one file per appendix
bib/                core.bib (main references) + extra.bib (reading list, tools)
scripts/            measure_workload.py
tools/              check_chapter.sh
```
