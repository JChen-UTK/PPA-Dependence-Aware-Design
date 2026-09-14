"""Render Figure 1, the framework decision flow, as an image file.

Figure 1 is a TikZ diagram rather than a data plot, so its source is the
standalone `fig1_framework.tex` next to this script: the same tikzset block and
tikzpicture the manuscript used, with the fonts the paper compiles with. This
program compiles that source and rasterises the result.

    python3 build_fig1_framework.py <output_dir> [source.tex]

`source.tex` may be either the standalone file (the default) or a manuscript that
still contains the tikzpicture, in which case the picture is extracted from it.

Output: "Fig 1. Framework decision flow.png" (600 dpi) and the same figure as a
vector PDF, which is what a journal prefers for line art when it accepts one.

Needs pdflatex on PATH and PyMuPDF for the rasterisation.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STEM = "Fig 1. Framework decision flow"
# Line art, so the raster is written at the 1000+ dpi publishers ask for
# rather than the 600 dpi used for the data figures.
DPI = 1200

# Preamble pieces the picture depends on, taken from the manuscript preamble.
# `txfonts` is what elsarticle's `times` class option loads, so the lettering
# matches the compiled paper.
PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{txfonts}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\newcommand{\Fix}{\mathrm{Fix}}
\newcommand{\AsG}{\mathrm{AsG}}
\newcommand{\AsC}{\mathrm{AsC}}
"""


def extract(tex: str) -> tuple[str, str]:
    """Return (tikzset block, tikzpicture block) from the manuscript source."""
    m = re.search(r"\\tikzset\{.*?\n\}\n", tex, re.S)
    if not m:
        raise SystemExit("could not find the \\tikzset block in the manuscript")
    tikzset = m.group(0)
    m = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, re.S)
    if not m:
        raise SystemExit("could not find a tikzpicture in the manuscript")
    return tikzset, m.group(0)


DEFAULT_SOURCE = Path(__file__).resolve().parent / "fig1_framework.tex"


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[1]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) == 3 else DEFAULT_SOURCE
    if not source.exists():
        raise SystemExit(f"source not found: {source}")

    text = source.read_text(encoding="utf-8")
    if "\\documentclass" in text and "\\begin{document}" in text:
        document = text                      # already a standalone figure source
    else:
        tikzset, picture = extract(text)     # a manuscript that still holds the picture
        document = PREAMBLE + tikzset + "\\begin{document}\n" + picture + "\n\\end{document}\n"

    with tempfile.TemporaryDirectory(prefix="fig1_") as tmp:
        work = Path(tmp)
        (work / "fig1.tex").write_text(document, encoding="utf-8")
        run = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig1.tex"],
            cwd=work, capture_output=True, text=True,
        )
        pdf = work / "fig1.pdf"
        if not pdf.exists():
            sys.stderr.write(run.stdout[-3000:])
            raise SystemExit("pdflatex did not produce a PDF")

        import pymupdf

        page = pymupdf.open(pdf)[0]
        width_in = page.rect.width / 72
        page.get_pixmap(dpi=DPI).save(out_dir / f"{STEM}.png")
        shutil.copy2(pdf, out_dir / f"{STEM}.pdf")

    png = out_dir / f"{STEM}.png"
    print(f"wrote {png.name}  ({width_in:.2f} in wide, {DPI} dpi)")
    print(f"wrote {STEM}.pdf (vector)")


if __name__ == "__main__":
    main()
