r"""Rebuild every manuscript figure into one directory.

Each of the four producers below writes PNG files whose names are exactly the
`\includegraphics` keys used by the manuscript, so the output directory can be
copied straight over `PPA_Manuscript/images/`.

Figure 1 is a TikZ diagram; its source is `fig1_framework.tex` in this folder.

    python3 make_all_figures.py <output_dir>

Interpreter: a Python with pandas, numpy and matplotlib (this project used
/opt/miniconda3/envs/py-3.13/bin/python3).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent                      # Code_Submission

# producer script -> the figures it writes
# build_figs_2_3_5_redesign.py now owns Figures 2-5. rebuild_figs_2_5_labels.py
# is kept as the record of how the superseded notebook versions were made, but it
# is no longer run here: it would write its own Figures 2-5 over these.
PRODUCERS = [
    ("build_figs_2_3_5_redesign.py", "Figures 2, 3, 4 and 5",
     ["--code-root", str(CODE_ROOT), "--out-dir", "{out}"]),
    ("build_print_figures.py", "Figures 6, 7 and A4-A10",
     ["{out}", str(CODE_ROOT)]),
    ("generate_figures_A1_A3_print.py", "Figures A1-A3",
     ["--data-dir", str(CODE_ROOT / "Input data and files" / "PJM Data"),
      "--output-dir", "{out}", "--dpi", "600"]),
]

EXPECTED = [
    "Fig 1. Framework decision flow.png",
    "Fig 2. PPA structure shares.png",
    "Fig 3. Severe structure transitions.png",
    "Fig 4. Contract term and delivery changes.png",
    "Fig 5. Seller exposure and buyer slack.png",
    "Fig 6. Fixed-Volume quantity changes.png",
    "Fig 7. Strike-price changes under risk aversion.png",
    "Fig A1. Buyer load profiles.png",
    "Fig A2. Seller generation profiles.png",
    "Fig A3. Buyer-side LMP profiles.png",
    "Fig A4. Outcome stability and PPA selection.png",
    "Fig A5. Contract term sensitivity.png",
    "Fig A6. PPA structure share sensitivity.png",
    "Fig A7. Delivered-volume changes under risk aversion.png",
    "Fig A8. Seller exposure changes under risk aversion.png",
    "Fig A9. Buyer exposure changes under risk aversion.png",
    "Fig A10. Buyer participation slack changes.png",
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("--- build_fig1_framework.py: Figure 1")
    subprocess.run([sys.executable, str(HERE / "build_fig1_framework.py"), str(out)],
                   check=True, cwd=str(HERE))

    for script, what, args in PRODUCERS:
        print(f"--- {script}: {what}")
        cmd = [sys.executable, str(HERE / script)] + [a.format(out=str(out)) for a in args]
        subprocess.run(cmd, check=True, cwd=str(HERE))

    missing = [f for f in EXPECTED if not (out / f).exists()]
    extra = sorted(p.name for p in out.glob("*.png") if p.name not in EXPECTED)
    print(f"\n{len(EXPECTED) - len(missing)}/{len(EXPECTED)} manuscript figures written to {out}")
    for f in missing:
        print("  MISSING:", f)
    for f in extra:
        print("  extra (not a manuscript figure):", f)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
