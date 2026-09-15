#!/usr/bin/env python3
"""Generate manuscript Appendix Figures A1, A2, and A3 from PJM CSV data.

The script consolidates the plotting logic originally split across:

* ``Demand Processing.ipynb`` (Figure A1),
* ``Generation Processing.ipynb`` (Figure A2), and
* ``LMP Processing.ipynb`` (Figure A3).

Required inputs, resolved relative to this script by default:

* ``Demand/hourly_metered_load_all.csv``
* ``Generation/hourly_total_gen_all_area.csv``
* ``Demand_LMP/OVEC_DOEX530.csv``

The published Figure A3 uses the OVEC ``LOAD`` pnode file, a buyer-side (load)
zone, which is what the supplement caption describes; the file name once said
"Seller-side" and was corrected. This script preserves the figure's actual
source and validates that provenance.

Exact raster reproduction was validated with pandas 2.2.3 and Matplotlib
3.10.7.  Other library versions can change font rasterization or PNG metadata
without changing the plotted statistics.

Each panel reports the mean, median, 5th percentile, and 95th percentile by
meteorological season and Eastern Prevailing Time hour.  The labels Q1--Q4
follow the original notebooks: Q1 = Dec--Feb, Q2 = Mar--May,
Q3 = Jun--Aug, and Q4 = Sep--Nov.

Example
-------
Run from any directory:

    python generate_figures_A1_A3_print.py

By default, figures are written to ``generated_figures_A1_A3`` beside this
script.  Use ``--output-dir`` to select another location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import figure_palette as PAL
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'axes.linewidth':0.6,'lines.linewidth':1.0})
import pandas as pd


SEASONS: Final[tuple[str, ...]] = ("Q1", "Q2", "Q3", "Q4")
HOURS: Final[tuple[int, ...]] = tuple(range(24))
PLOT_FONT_SIZE: Final[int] = 6
MONTH_TO_SEASON: Final[dict[int, str]] = {
    12: "Q1",
    1: "Q1",
    2: "Q1",
    3: "Q2",
    4: "Q2",
    5: "Q2",
    6: "Q3",
    7: "Q3",
    8: "Q3",
    9: "Q4",
    10: "Q4",
    11: "Q4",
}


@dataclass(frozen=True)
class FigureSpec:
    """Plotting settings that differ among Figures A1--A3."""

    output_name: str
    entity_label: str
    entity_value: str
    y_label: str
    box_color: str
    mean_color: str


FIGURE_A1: Final[FigureSpec] = FigureSpec(
    output_name="Fig A1. Buyer load profiles.png",
    entity_label="Load Area",
    entity_value="DUQ",
    y_label="Load (MW)",
    box_color=PAL.PROFILE_FILL["demand"],
    mean_color=PAL.PROFILE_MEAN,
)
FIGURE_A2: Final[FigureSpec] = FigureSpec(
    output_name="Fig A2. Seller generation profiles.png",
    entity_label="Area",
    entity_value="MIDATL",
    y_label="Generation (MW)",
    box_color=PAL.PROFILE_FILL["generation"],
    mean_color=PAL.PROFILE_MEAN,
)
FIGURE_A3: Final[FigureSpec] = FigureSpec(
    output_name="Fig A3. Buyer-side LMP profiles.png",
    entity_label="Zone",
    entity_value="OVEC",
    y_label="LMP ($/MWh)",
    box_color=PAL.PROFILE_FILL["lmp"],
    mean_color=PAL.PROFILE_MEAN,
)


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    """Raise a clear error when an input schema does not match expectations."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def parse_ept(series: pd.Series, source: Path) -> pd.Series:
    """Parse the two timestamp formats used in the frozen PJM inputs."""

    parsed = pd.to_datetime(series, format="%m/%d/%y %H:%M", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            series.loc[remaining],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )

    bad_count = int(parsed.isna().sum())
    if bad_count:
        raise ValueError(
            f"{source} contains {bad_count} unparseable "
            "datetime_beginning_ept value(s)."
        )
    return parsed


def summarize_by_season_and_hour(
    frame: pd.DataFrame,
    value_column: str,
    source: Path,
) -> pd.DataFrame:
    """Compute the statistics encoded by the original notebook box plots."""

    working = frame[["datetime_beginning_ept", value_column]].copy()
    working["datetime"] = parse_ept(working["datetime_beginning_ept"], source)
    working["hour"] = working["datetime"].dt.hour
    working["season"] = working["datetime"].dt.month.map(MONTH_TO_SEASON)

    if working[value_column].isna().any():
        missing = int(working[value_column].isna().sum())
        raise ValueError(f"{source} contains {missing} missing {value_column} value(s).")

    grouped = working.groupby(["season", "hour"], observed=True)[value_column]
    summary = grouped.agg(
        mean="mean",
        median="median",
        p5=lambda values: values.quantile(0.05),
        p95=lambda values: values.quantile(0.95),
        count="count",
    ).reindex(pd.MultiIndex.from_product([SEASONS, HOURS], names=["season", "hour"]))

    incomplete = summary[summary.isna().any(axis=1)]
    if not incomplete.empty:
        groups = [f"{season}/H{hour}" for season, hour in incomplete.index]
        raise ValueError(f"{source} lacks observations for: {', '.join(groups)}")
    return summary


def load_demand(source: Path) -> pd.DataFrame:
    """Load DUQ demand without filtering the notebook's is_verified field."""

    frame = pd.read_csv(source)
    require_columns(
        frame,
        {"datetime_beginning_ept", "load_area", "mw", "is_verified"},
        source,
    )
    selected = frame.loc[frame["load_area"].eq(FIGURE_A1.entity_value)].copy()
    if selected.empty:
        raise ValueError(f"{source} contains no {FIGURE_A1.entity_value} load rows.")
    return selected


def load_generation(source: Path) -> pd.DataFrame:
    """Load MIDATL generation and reproduce the notebook's zero clipping."""

    frame = pd.read_csv(source)
    require_columns(
        frame,
        {
            "datetime_beginning_ept",
            "area",
            "solar_generation_mw",
            "wind_generation_mw",
        },
        source,
    )
    selected = frame.loc[frame["area"].eq(FIGURE_A2.entity_value)].copy()
    if selected.empty:
        raise ValueError(f"{source} contains no {FIGURE_A2.entity_value} generation rows.")

    # The original generation notebook clips the two components before summing.
    selected["figure_total_generation_mw"] = (
        selected["solar_generation_mw"].clip(lower=0)
        + selected["wind_generation_mw"].clip(lower=0)
    )
    return selected


def load_lmp(source: Path) -> pd.DataFrame:
    """Load the OVEC LOAD-pnode LMP series used by the published Figure A3."""

    frame = pd.read_csv(source)
    require_columns(
        frame,
        {"datetime_beginning_ept", "zone", "type", "total_lmp_rt"},
        source,
    )
    selected = frame.loc[frame["zone"].eq(FIGURE_A3.entity_value)].copy()
    if selected.empty:
        raise ValueError(f"{source} contains no {FIGURE_A3.entity_value} LMP rows.")

    node_types = set(selected["type"].dropna().astype(str).str.upper())
    if node_types != {"LOAD"}:
        raise ValueError(
            f"{source} was expected to contain only LOAD rows for OVEC; "
            f"found {sorted(node_types)}."
        )
    return selected


def plot_summary(summary: pd.DataFrame, spec: FigureSpec, destination: Path, dpi: int) -> None:
    """Render one four-panel figure using the original notebook styling."""

    plt.style.use("default")
    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.6, "lines.linewidth": 0.8})
    figure, axes = plt.subplots(2, 2, figsize=(522 / 72.27, 4.4), sharey=True)

    for index, (axis, season) in enumerate(zip(axes.flat, SEASONS, strict=True)):
        seasonal = summary.loc[season]
        box_stats = [
            {
                "med": seasonal.loc[hour, "median"],
                "q1": seasonal.loc[hour, "p5"],
                "q3": seasonal.loc[hour, "p95"],
                "whislo": seasonal.loc[hour, "p5"],
                "whishi": seasonal.loc[hour, "p95"],
                "fliers": [],
            }
            for hour in HOURS
        ]

        axis.bxp(
            box_stats,
            positions=HOURS,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            # The notebooks passed both ``facecolor='lightblue'`` and
            # ``color=<figure color>``. Matplotlib's ``color`` property
            # overrides both face and edge colors, so the explicit equivalent
            # below reproduces the output without emitting a warning.
            boxprops={"facecolor": spec.box_color, "edgecolor": spec.box_color},
            medianprops={"color": "black", "linewidth": 2},
        )
        axis.scatter(
            HOURS,
            seasonal["mean"].to_numpy(),
            color=spec.mean_color,
            label="Mean",
            zorder=3,
            s=25,
            edgecolors="white",
            linewidths=0.5,
        )
        axis.set_xticks(HOURS)
        axis.set_xlabel("Hour", fontsize=PLOT_FONT_SIZE)
        if index % 2 == 0:
            axis.set_ylabel(spec.y_label, fontsize=PLOT_FONT_SIZE)
        axis.set_title(
            f"{season} - {spec.entity_label}: {spec.entity_value}",
            fontsize=PLOT_FONT_SIZE,
        )
        axis.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
        axis.legend(fontsize=PLOT_FONT_SIZE)
        axis.grid(True, linestyle="--", alpha=0.7)

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _ax in axes.ravel():
        _ax.set_xticks(range(0, 24, 4)); _ax.set_xticklabels([str(h) for h in range(0, 24, 4)])
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def positive_integer(value: str) -> int:
    """Argparse converter for a positive integer DPI."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser(script_dir: Path) -> argparse.ArgumentParser:
    """Construct the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir.parent / "Input data and files" / "PJM Data",
        help="PJM Data directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "generated_figures_A1_A3",
        help="Directory for the three PNG files.",
    )
    parser.add_argument(
        "--dpi",
        type=positive_integer,
        default=600,
        help="PNG resolution in dots per inch (default: 100).",
    )
    return parser


def main() -> None:
    """Load the frozen CSV inputs and generate all three appendix figures."""

    script_dir = Path(__file__).resolve().parent
    args = build_parser(script_dir).parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    demand_source = data_dir / "Demand" / "hourly_metered_load_all.csv"
    generation_source = data_dir / "Generation" / "hourly_total_gen_all_area.csv"
    lmp_source = data_dir / "Demand_LMP" / "OVEC_DOEX530.csv"
    for source in (demand_source, generation_source, lmp_source):
        if not source.is_file():
            raise FileNotFoundError(f"Required input not found: {source}")

    demand_summary = summarize_by_season_and_hour(load_demand(demand_source), "mw", demand_source)
    generation_summary = summarize_by_season_and_hour(
        load_generation(generation_source),
        "figure_total_generation_mw",
        generation_source,
    )
    lmp_summary = summarize_by_season_and_hour(load_lmp(lmp_source), "total_lmp_rt", lmp_source)

    outputs = (
        (demand_summary, FIGURE_A1),
        (generation_summary, FIGURE_A2),
        (lmp_summary, FIGURE_A3),
    )
    for summary, spec in outputs:
        destination = output_dir / spec.output_name
        plot_summary(summary, spec, destination, args.dpi)
        print(f"Generated: {destination}")


if __name__ == "__main__":
    main()
