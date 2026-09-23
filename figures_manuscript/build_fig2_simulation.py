"""Render Figure 2, the simulation framework flow, as an image file.

Figure 2 is a TikZ diagram like Figure 1, so it is compiled and rasterised by the
same procedure as `build_fig1_framework.py`; only the source and the output name
differ. The source is `fig2_simulation.tex` next to this script.

    python3 build_fig2_simulation.py <output_dir> [source.tex]

Output: "Fig 2. Simulation framework.png" (1200 dpi) and the same figure as a
vector PDF, which is what the manuscript includes.

Needs pdflatex on PATH and PyMuPDF for the rasterisation.
"""
from __future__ import annotations

from pathlib import Path

import build_fig1_framework as tikz_figure

tikz_figure.STEM = "Fig 2. Simulation framework"
tikz_figure.DEFAULT_SOURCE = Path(__file__).resolve().parent / "fig2_simulation.tex"

if __name__ == "__main__":
    tikz_figure.main()
