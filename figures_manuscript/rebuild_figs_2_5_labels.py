"""Rebuild manuscript Figures 2-5 from the Code_Submission notebook with the
manuscript's canonical axis labels.

The notebook `Code_Submission/simulation_mutation/
plot_contract_price_volume_utility_minimal_update.ipynb` produces the four
figures. This script reads that notebook read-only, applies the label
substitutions below, and runs the code in a temporary working directory so
nothing in Code_Submission is written.

Label changes (2026-09-10):
  fig_contract_terms_by_family   "Fixed volume"      -> "Fixed-Volume quantity"
  fig_contract_terms_by_family   "M ean delivered"   -> "Mean delivered"   (typo)
  fig_objective_slack_movement   "Buyer slack"       -> "Buyer participation slack"
  figs 2 and 4 x axis            "Correlation shift" -> "Requested correlation shift"

Figure 3 comes out byte-identical to the shipped PNG, which is the check that the
rebuild reproduces the original run.

Usage:
  python3 rebuild_figs_2_5_labels.py --code-root <Code_Submission> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NOTEBOOK = "simulation_mutation/plot_contract_price_volume_utility_minimal_update.ipynb"
INPUT_CSV = "simulation_mutation/Output files (Risk Neutral, Mutation, Verified)/Simulation_Plot_Data_All_Matches.csv"
OUT_SUBDIR = "Output files (Risk Neutral, Mutation, Verified)/Contract_Price_Volume_Utility_Changes/journal_figures"

# (old, new, expected number of occurrences in the notebook's code)
SUBSTITUTIONS = [
    ('"Δ Fixed volume\\n(MW; Fix→Fix)"', '"Δ Fixed-Volume quantity\\n(MW; Fix→Fix)"', 1),
    ('"Δ M ean delivered\\nvolume (MW)"', '"Δ Mean delivered\\nvolume (MW)"', 1),
    ('"Δ Buyer slack"', '"Δ Buyer participation slack"', 1),
    # Figures 2 and 4; matches "Requested correlation shift" in Figs 6, 7 and A5-A10.
    # Three occurrences in the style dicts plus two fallbacks in the plotting helper.
    ('"Correlation shift"', '"Requested correlation shift"', 5),
]

FIGURE_NAMES = {
    "fig_profile_share_by_family.png": "Fig 2. PPA structure shares.png",
    "fig_transition_matrices_by_family_severe.png": "Fig 3. Severe structure transitions.png",
    "fig_contract_terms_by_family.png": "Fig 4. Contract term and delivery changes.png",
    "fig_objective_slack_movement.png": "Fig 5. Seller exposure and buyer slack.png",
}


def build_script(notebook: Path, work_dir: Path) -> str:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    override = f'simulation_mutation_dir_override = r"{work_dir / "simulation_mutation"}"'
    pairs = [("simulation_mutation_dir_override = None", override, 1)] + SUBSTITUTIONS
    for old, new, expected in pairs:
        found = code.count(old)
        if found != expected:
            raise SystemExit(f"expected {expected} occurrence(s) of {old!r}, found {found}")
        code = code.replace(old, new)
    # Figure 4 names no colour and takes the first entry of the default
    # cycle, so the cycle is set here to keep it on the paper's palette.
    preamble = (
        'import matplotlib\n'
        'matplotlib.use("Agg")\n'
        'import sys\n'
        f'sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n'
        'import matplotlib.pyplot as plt\n'
        'import figure_palette as PAL\n'
        'PAL.apply_palette(plt)\n'
    )
    return preamble + code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", default=str(Path(__file__).resolve().parent.parent),
                    help="path to Code_Submission (default: the parent of this script's folder)")
    ap.add_argument("--out-dir", required=True, help="directory to write the renamed PNGs into")
    args = ap.parse_args()

    root = Path(args.code_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="figs25_") as tmp:
        work = Path(tmp)
        # A real copy, not a symlink: the notebook resolves the input path, and a
        # symlink would make it write its outputs back into Code_Submission.
        dest = work / Path(INPUT_CSV)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / INPUT_CSV, dest)

        script = work / "run_figs.py"
        script.write_text(build_script(root / NOTEBOOK, work), encoding="utf-8")
        subprocess.run([sys.executable, str(script)], cwd=work, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        produced = work / "simulation_mutation" / OUT_SUBDIR
        for stem, manuscript_name in FIGURE_NAMES.items():
            src = produced / stem
            if not src.exists():
                raise SystemExit(f"missing expected figure: {src}")
            shutil.copy2(src, out_dir / manuscript_name)
            print(f"wrote {manuscript_name}")


if __name__ == "__main__":
    main()
