"""Redesigned Figures 2, 3 and 5 at printed width.

Each figure answers a different question, so each uses a different chart family.

  Figure 2  What the text quotes is a CHANGE: "the Fixed-Volume share increases
            by 5.6%, 9.5% and 9.5% of the configurations". The shipped figure
            plots levels in a 2x2 line grid, so the reader has to subtract.
            Redrawn as one horizontal diverging bar chart: twelve rows, one per
            channel and shift size, three bars per row, zero = the reference.
  Figure 3  What the text reads off it is a FLOW: "mainly through
            As-Consumed-to-Fixed transitions", "the larger gross movement out of
            Fixed-Volume is toward As-Consumed". Redrawn as a 2x2 of alluvial
            diagrams, ribbon width = number of configurations moving.
  Figure 5  The only figure about the JOINT seller-buyer outcome; Figure 4
            already carries the marginal distributions. Kept as a scatter in a
            2x2, but the four panels now share one symmetric-log axis pair in
            $ million instead of four different linear ranges with a 1e8
            multiplier, and each quadrant carries the exact share of
            configurations in it.

No plotted value is changed. Every number is read from the journal tables the
shipped figures are built from, and `--verify` re-derives the published
reference counts from them.

Colours are Okabe-Ito for the contract structures and a grey/crimson/indigo set
for the shift sizes; both were checked for dichromat separation in OKLab.

Usage:
  python3 build_figs_2_3_5_redesign.py [--out-dir DIR] [--code-root DIR]
                                       [--also-copy-to DIR] [--verify]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, PercentFormatter

from figure_palette import (ANNOT_FS, BASE_FS, CHANNEL_COLOR, CHANNEL_MARKER, GRID,
                            INK, MUTED, SHIFT_COLOR, STRUCTURE_COLOR, fit_legend,
                            house_rcparams, tidy)

# ----------------------------------------------------------------------------
# Constants shared with the manuscript
# ----------------------------------------------------------------------------

# \textwidth of the elsarticle 5p twocolumn class, measured in the build: 522 pt
# of 1/72.27 in, which is 7.2229 in. Draw at that width and include the figure
# at width=\textwidth so nothing is rescaled and the drawn point sizes hold.
PRINT_WIDTH_IN = 522.0 / 72.27
DPI = 600

TABLE_SUBDIR = (
    "simulation_mutation/Output files (Risk Neutral, Mutation, Verified)/"
    "Contract_Price_Volume_Utility_Changes/journal_tables"
)

CHANNEL_ORDER = [
    "Generation-load mismatch",
    "Seller-buyer nodal price decoupling",
    "Buyer load-price intensification",
    "Seller generation-price cannibalization",
]

# Two-line panel titles; the en dashes match the manuscript's prose.
CHANNEL_WRAPPED = {
    "Generation-load mismatch": "Generation–load\nmismatch",
    "Seller-buyer nodal price decoupling": "Seller–buyer nodal\nprice decoupling",
    "Buyer load-price intensification": "Buyer load–price\nintensification",
    "Seller generation-price cannibalization": "Seller generation–price\ncannibalization",
}

STRUCTURE_ORDER = ["Fix", "AsC", "AsG"]
STRUCTURE_LONG = {
    "Fix": "Fixed-Volume",
    "AsC": "As-Consumed",
    "AsG": "As-Generated",
}

# Figure 3 writes the structure beside a narrow stack block, so it keeps the
# abbreviation there; Figure 2 has room in its panel titles and spells it out.

SHIFT_LEVELS = [0.1, 0.3, 0.5]
SHIFT_NAME = {0.1: "mild", 0.3: "moderate", 0.5: "severe"}

N_CONFIG = 126




def set_style() -> None:
    plt.rcParams.update(house_rcparams())


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_tables(code_root: Path) -> dict[str, pd.DataFrame]:
    tdir = code_root / TABLE_SUBDIR
    if not tdir.is_dir():
        raise SystemExit(f"journal_tables not found under {tdir}")
    return {
        "shares": pd.read_csv(tdir / "Journal_Profile_Shares_ForPlot.csv"),
        "severe": pd.read_csv(tdir / "Journal_Transition_Summary_Severe_ByFamily.csv"),
        "long": pd.read_csv(
            tdir / "Journal_Long_Change_By_MatchScenario.csv",
            usecols=[
                "match_id", "mutation_family_normalized", "target_shift_signed",
                "target_shift_abs", "baseline_selected_contract_type",
                "selected_contract_type", "delta_seller_metric",
                "delta_buyer_participation_slack", "baseline_seller_metric_value",
                "delta_strike_price_mwh", "delta_fixed_volume_mw",
                "delta_mean_delivered_volume_proxy_mw",
            ],
            low_memory=False,
        ),
    }


def share_changes(shares: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Change in selected share from the reference, in percentage points."""
    out: dict[str, dict[str, list[float]]] = {}
    for channel in CHANNEL_ORDER:
        sub = shares[shares.mutation_family_normalized.eq(channel)]
        ref = {s: 0.0 for s in STRUCTURE_ORDER}
        for _, r in sub[sub.is_baseline_reference].iterrows():
            ref[r.selected_contract_type] = float(r.share)
        per_structure: dict[str, list[float]] = {s: [] for s in STRUCTURE_ORDER}
        for lv in SHIFT_LEVELS:
            rows = sub[~sub.is_baseline_reference & np.isclose(sub.target_shift_abs, lv)]
            level = {s: 0.0 for s in STRUCTURE_ORDER}
            for _, r in rows.iterrows():
                level[r.selected_contract_type] = float(r.share)
            for s in STRUCTURE_ORDER:
                per_structure[s].append((level[s] - ref[s]) * 100.0)
        out[channel] = per_structure
    return out


def verify(tables: dict[str, pd.DataFrame]) -> None:
    """Re-derive published numbers from the same tables the figures read."""
    print("Verification against the manuscript (Sections 5.1 and 5.2)")
    long = tables["long"]
    base = long.drop_duplicates("match_id")

    counts = base["baseline_selected_contract_type"].value_counts()
    print(f"  reference selections: Fix {counts.get('Fix', 0)} "
          f"({counts.get('Fix', 0) / len(base):.1%}), "
          f"AsC {counts.get('AsC', 0)} ({counts.get('AsC', 0) / len(base):.1%}), "
          f"n = {len(base)}   [manuscript: 87 (69.0%), 39 (31.0%), 126]")

    med = base.groupby("baseline_selected_contract_type")["baseline_seller_metric_value"].median() / 1e6
    print(f"  median reference seller exposure index: Fix ${med['Fix']:,.0f} million, "
          f"AsC ${med['AsC']:,.0f} million   [manuscript: -$1,584 million, -$1,137 million]")

    up = long.groupby("mutation_family_normalized")["delta_seller_metric"].apply(lambda s: (s > 0).mean())
    print("  share of configurations with a rising seller exposure index:")
    for ch in CHANNEL_ORDER:
        print(f"    {ch:<42s} {up[ch]:6.1%}")
    print("    [manuscript: mismatch and decoupling move it up, the other two more often down]")

    delta = share_changes(tables["shares"])
    quoted = [
        ("Generation-load mismatch", "Fix", "+5.6, +9.5, +9.5"),
        ("Seller-buyer nodal price decoupling", "Fix", "+4.8, +9.5, +14.3"),
        ("Buyer load-price intensification", "AsC", "+3.2, +25.4, +29.4"),
        ("Seller generation-price cannibalization", "AsG", "0, +8.7, +12.7"),
    ]
    print("  change in selected share, in percentage points of the 126 configurations:")
    for ch, st, expected in quoted:
        got = ", ".join(f"{v:+.1f}" for v in delta[ch][st])
        print(f"    {st} under {ch:<42s} {got}   [manuscript: {expected}]")
    print()


def signed_shift(shares: pd.DataFrame, channel: str) -> float:
    sub = shares[shares.mutation_family_normalized.eq(channel) & ~shares.is_baseline_reference]
    return float(np.sign(sub.target_shift_signed.iloc[0]))


def shift_label(sign: float, magnitude: float) -> str:
    return f"{'+' if sign > 0 else '−'}{magnitude:.2f}"


# ----------------------------------------------------------------------------
# Figure 2 — how far each selected share moves off the reference
# ----------------------------------------------------------------------------

# The requested shift is an ordered dose (reference, mild, moderate, severe), so
# a connected profile is the right mark: it carries the progression that a bar
# chart throws away. What changes from the shipped figure is the quantity and the
# faceting. The panel is the contract structure and the profile is the channel,
# so the comparison the Results text keeps making -- "the same substitution
# toward Fixed-Volume" -- happens inside one panel instead of across four.
#
# Channels carry their own colour here. Telling four same-coloured profiles apart
# by dash pattern alone did not work at this panel size. These four are an
# Okabe-Ito subset, every pair clear of the dichromat floor, with a distinct
# marker as a second cue. Colour therefore means channel in this figure and
# contract structure in Figure 3; each figure carries its own key, and the panel
# titles here name the structures.


def build_fig2(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    shares = tables["shares"]
    delta = share_changes(shares)

    fig = plt.figure(figsize=(PRINT_WIDTH_IN, 3.00))
    left, right, bottom, top, gap = 0.098, 0.995, 0.165, 0.770, 0.026
    width = (right - left - 2 * gap) / 3
    axes = [fig.add_axes((left + i * (width + gap), bottom, width, top - bottom))
            for i in range(3)]

    x = np.arange(4)          # the reference, then the three shift sizes

    for ai, (ax, st) in enumerate(zip(axes, STRUCTURE_ORDER)):
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=3)
        for channel in CHANNEL_ORDER:
            color, marker = CHANNEL_COLOR[channel], CHANNEL_MARKER[channel]
            y = np.concatenate(([0.0], delta[channel][st]))
            ax.plot(x, y, marker=marker, markersize=3.6, linewidth=1.4,
                    color=color, markeredgecolor="white", markeredgewidth=0.6,
                    clip_on=False, zorder=4)

        ax.set_title(STRUCTURE_LONG[st], fontsize=BASE_FS, color=INK, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(["Ref."] + [f"{m:.2f}" for m in SHIFT_LEVELS])
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(-32, 32)
        ax.set_yticks(np.arange(-30, 31, 10))
        # The values are already in percentage points, so the tick carries the
        # unit and the axis title does not have to spell it out.
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ai:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

    axes[0].set_ylabel("Change in selected share\nfrom the reference")
    fig.text((left + right) / 2, 0.035, "Shift size",
             ha="center", va="bottom", fontsize=BASE_FS)

    # The x axis shows the shift magnitude, so the requested direction is not in
    # the figure: it is fixed per channel where the channels are defined, and the
    # caption is the place to restate it if a reader needs it here.
    handles = [Line2D([], [], marker=CHANNEL_MARKER[c], linestyle="-",
                      color=CHANNEL_COLOR[c], markeredgecolor="white",
                      markeredgewidth=0.6, markersize=3.6, linewidth=1.4,
                      label=CHANNEL_WRAPPED[c].replace(chr(10), " "))
               for c in CHANNEL_ORDER]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left - 0.004, 0.885),
                     bbox_transform=fig.transFigure, ncol=2, columnspacing=1.6, handletextpad=0.5)
    fit_legend(fig, leg)

    return save(fig, out_dir, "Fig 2. PPA structure shares")


# ----------------------------------------------------------------------------
# Figure 3 — alluvial flows: which configurations move, and where to
# ----------------------------------------------------------------------------

def ribbon(ax, x0, x1, y0_lo, y0_hi, y1_lo, y1_hi, color, alpha):
    """A flow band between two stack segments, with a smoothstep centreline."""
    t = np.linspace(0.0, 1.0, 120)
    s = 3 * t ** 2 - 2 * t ** 3
    ax.fill_between(x0 + (x1 - x0) * t,
                    y0_lo + (y1_lo - y0_lo) * s,
                    y0_hi + (y1_hi - y0_hi) * s,
                    facecolor=color, alpha=alpha, linewidth=0, zorder=2)


def _fit_x_to_content(fig, axes, pad_frac: float = 0.008, passes: int = 3) -> None:
    """Shrink each axis to the horizontal extent actually drawn in it.

    The block labels sit outside the stacks and their width is fixed in points,
    so their extent in data units depends on the x range that is being chosen.
    Guessing it left a wide blank band between the two columns, so measure the
    rendered artists and iterate: each pass reads the true extent under the
    current range and re-fits, which settles in two or three rounds.
    """
    from matplotlib.transforms import Bbox

    for _ in range(passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            boxes = []
            for artist in list(ax.texts) + list(ax.patches) + list(ax.collections):
                try:
                    extent = artist.get_window_extent(renderer)
                except Exception:
                    continue
                if extent.width > 0:
                    boxes.append(extent)
            if not boxes:
                continue
            union = Bbox.union(boxes)
            inv = ax.transData.inverted()
            x0 = inv.transform((union.x0, union.y0))[0]
            x1 = inv.transform((union.x1, union.y1))[0]
            pad = pad_frac * (x1 - x0)
            ax.set_xlim(x0 - pad, x1 + pad)


def build_fig3(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    severe = tables["severe"]

    fig = plt.figure(figsize=(PRINT_WIDTH_IN, 3.90))
    left, right, bottom, top = 0.018, 0.992, 0.028, 0.938
    col_gap, row_gap = 0.032, 0.092
    # Number marks are set a little larger than elsewhere: this figure
    # carries its counts inside the artwork rather than on an axis.
    label_fs, thin_fs = ANNOT_FS + 0.5, ANNOT_FS - 0.5
    pw = (right - left - col_gap) / 2
    ph = (top - bottom - row_gap) / 2
    axes = [fig.add_axes((left + (i % 2) * (pw + col_gap),
                          bottom + (1 - i // 2) * (ph + row_gap), pw, ph))
            for i in range(4)]

    x_src, x_dst, bar_w = 0.0, 0.845, 0.085
    # Height of one y data unit in points, for the ribbon-label fit test.
    pt_per_unit = ph * fig.get_figheight() * 72.0 / 1.130
    block_gap = 0.045                      # vertical gap between stack blocks

    def stack(types, totals):
        usable = 1.0 - block_gap * (len(types) - 1)
        spans, cursor = {}, 1.0
        for st in types:
            h = usable * totals[st] / N_CONFIG
            spans[st] = (cursor - h, cursor)
            cursor -= h + block_gap
        return spans

    for ax, channel in zip(axes, CHANNEL_ORDER):
        sub = severe[severe.mutation_family_normalized.eq(channel)]
        src_types = [st for st in STRUCTURE_ORDER
                     if st in set(sub.baseline_selected_contract_type)]
        dst_types = [st for st in STRUCTURE_ORDER
                     if st in set(sub.selected_contract_type)]
        src_tot = {st: int(sub[sub.baseline_selected_contract_type.eq(st)].n.sum())
                   for st in src_types}
        dst_tot = {st: int(sub[sub.selected_contract_type.eq(st)].n.sum())
                   for st in dst_types}
        src_span, dst_span = stack(src_types, src_tot), stack(dst_types, dst_tot)

        for st, (lo, hi) in src_span.items():
            ax.add_patch(plt.Rectangle((x_src, lo), bar_w, hi - lo,
                                       facecolor=STRUCTURE_COLOR[st], linewidth=0, zorder=3))
            ax.text(x_src - 0.028, (lo + hi) / 2, f"{st}\n{src_tot[st]}", ha="right",
                    va="center", fontsize=label_fs, color=INK, linespacing=1.2, zorder=4)
        for st, (lo, hi) in dst_span.items():
            ax.add_patch(plt.Rectangle((x_dst, lo), bar_w, hi - lo,
                                       facecolor=STRUCTURE_COLOR[st], linewidth=0, zorder=3))
            ax.text(x_dst + bar_w + 0.028, (lo + hi) / 2, f"{st}\n{dst_tot[st]}", ha="left",
                    va="center", fontsize=label_fs, color=INK, linespacing=1.2, zorder=4)

        # Ribbons leave each source block top-down in destination order and enter
        # each destination block top-down in source order, which is what keeps an
        # alluvial diagram from crossing itself unnecessarily.
        src_cursor = {st: src_span[st][1] for st in src_types}
        dst_cursor = {st: dst_span[st][1] for st in dst_types}
        for src in src_types:
            for dst in dst_types:
                row = sub[sub.baseline_selected_contract_type.eq(src)
                          & sub.selected_contract_type.eq(dst)]
                if not len(row):
                    continue
                n = int(row.n.iloc[0])
                h_src = (src_span[src][1] - src_span[src][0]) * n / src_tot[src]
                h_dst = (dst_span[dst][1] - dst_span[dst][0]) * n / dst_tot[dst]
                y0_hi, y1_hi = src_cursor[src], dst_cursor[dst]
                ribbon(ax, x_src + bar_w, x_dst, y0_hi - h_src, y0_hi,
                       y1_hi - h_dst, y1_hi, STRUCTURE_COLOR[src],
                       0.75 if src == dst else 0.42)
                # A count is written on the ribbon at the point along it where
                # the ribbon is thick enough to hold the text. Retained flows are
                # labelled mid-span; switching flows are labelled near the
                # destination, because the two switching ribbons of a channel
                # cross at mid-span and would stack their labels on one spot.
                t = 0.45 if src == dst else 0.82
                sm = 3 * t ** 2 - 2 * t ** 3
                hi_t = y0_hi + (y1_hi - y0_hi) * sm
                lo_t = (y0_hi - h_src) + ((y1_hi - h_dst) - (y0_hi - h_src)) * sm
                thick_pt = (hi_t - lo_t) * pt_per_unit
                if thick_pt >= thin_fs - 0.5:
                    ax.text(x_src + bar_w + (x_dst - x_src - bar_w) * t,
                            (hi_t + lo_t) / 2, str(n), ha="center", va="center",
                            color=INK, zorder=5,
                            fontsize=label_fs if thick_pt >= label_fs + 1.5 else thin_fs,
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=0.8, pad=0.5))
                src_cursor[src] -= h_src
                dst_cursor[dst] -= h_dst

        ax.text(bar_w / 2, 1.035, "Reference", ha="center", va="bottom",
                fontsize=ANNOT_FS - 0.5, color=MUTED, style="italic")
        ax.text(x_dst + bar_w / 2, 1.035, "After", ha="center", va="bottom",
                fontsize=ANNOT_FS - 0.5, color=MUTED, style="italic")
        ax.set_title(CHANNEL_WRAPPED[channel].replace(chr(10), " "),
                     fontsize=BASE_FS, color=INK, pad=4)
        ax.set_xlim(-0.210, 1.140)   # provisional; fitted below
        ax.set_ylim(-0.025, 1.105)
        ax.axis("off")

    _fit_x_to_content(fig, axes)

    return save(fig, out_dir, "Fig 3. Severe structure transitions")


# ----------------------------------------------------------------------------
# Figure 4 — contract-term and delivery responses, one colour per channel
# ----------------------------------------------------------------------------

# Each column is one dependence channel and wears that channel's colour, the
# same one it has as a curve in Figure 2, so a reader who has learnt the channel
# colours there keeps them here. Rows share a y axis: on the shipped figure every
# panel was scaled independently, which magnified a channel that barely responds
# into something that looked like a large effect.
FIG4_METRICS = [
    ("delta_strike_price_mwh", "Δ strike price\n(\\$/MWh)"),
    ("delta_fixed_volume_mw", "Δ Fixed-Volume quantity\n(MW; Fix→Fix)"),
    ("delta_mean_delivered_volume_proxy_mw", "Δ mean delivered\nvolume (MW)"),
]


def build_fig4(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    long = tables["long"]

    fig = plt.figure(figsize=(PRINT_WIDTH_IN, 5.70))
    left, right, bottom, top = 0.135, 0.995, 0.095, 0.905
    col_gap, row_gap = 0.018, 0.075
    pw = (right - left - 3 * col_gap) / 4
    ph = (top - bottom - 2 * row_gap) / 3
    x = np.arange(3)

    for r, (metric, ylabel) in enumerate(FIG4_METRICS):
        axes_row, lo, hi = [], np.inf, -np.inf
        for c, channel in enumerate(CHANNEL_ORDER):
            ax = fig.add_axes((left + c * (pw + col_gap),
                               bottom + (2 - r) * (ph + row_gap), pw, ph))
            axes_row.append(ax)
            sub = long[long.mutation_family_normalized.eq(channel)]
            colour = CHANNEL_COLOR[channel]
            med, ns, signs = [], [], []
            for lv in SHIFT_LEVELS:
                sel = sub[np.isclose(sub.target_shift_abs, lv)]
                v = pd.to_numeric(sel[metric], errors="coerce").dropna()
                ns.append(int(v.size))
                signs.append(float(np.sign(sel.target_shift_signed.iloc[0])))
                med.append(float(v.median()))
                q10, q25, q75, q90 = (float(v.quantile(q)) for q in (0.10, 0.25, 0.75, 0.90))
                i = len(med) - 1
                ax.vlines(x[i], q10, q90, color=colour, linewidth=0.8, zorder=3)
                ax.vlines(x[i], q25, q75, color=colour, linewidth=3.0, zorder=4)
                lo, hi = min(lo, q10), max(hi, q90)
            ax.plot(x, med, color=colour, linewidth=1.2, marker=CHANNEL_MARKER[channel],
                    markersize=3.2, markerfacecolor="white", markeredgecolor=colour,
                    markeredgewidth=0.9, zorder=5)
            ax.axhline(0, color=MUTED, linewidth=0.6, zorder=2)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{m:.2f}" for m in SHIFT_LEVELS])
            ax.set_xlim(-0.45, 2.45)
            ax.tick_params(axis="x", length=0, pad=2)
            tidy(ax)
            if r == 0:
                ax.set_title(CHANNEL_WRAPPED[channel], fontsize=BASE_FS,
                             color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=BASE_FS)

        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        for c, ax in enumerate(axes_row):
            ax.set_ylim(lo - pad, hi + pad)
            if c:
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)

    # The Fixed-Volume row is the only one whose count changes, so it carries the
    # n annotation and the other two rows say so once in the caption.
    for c, channel in enumerate(CHANNEL_ORDER):
        sub = long[long.mutation_family_normalized.eq(channel)]
        ax = fig.axes[4 + c]
        for i, lv in enumerate(SHIFT_LEVELS):
            sel = sub[np.isclose(sub.target_shift_abs, lv)]
            n = int(pd.to_numeric(sel["delta_fixed_volume_mw"], errors="coerce").notna().sum())
            ax.annotate(f"n={n}", xy=(x[i], 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -11), textcoords="offset points", ha="center",
                        va="top", fontsize=ANNOT_FS - 0.5, color=MUTED, annotation_clip=False)

    fig.text((left + right) / 2, 0.018, "Shift size",
             ha="center", va="bottom", fontsize=BASE_FS)
    handles = [Line2D([], [], color=MUTED, linewidth=0.8, label="10–90% range"),
               Line2D([], [], color=MUTED, linewidth=3.0, label="Interquartile range"),
               Line2D([], [], color=MUTED, linewidth=1.2, marker="o", markersize=3.2,
                      markerfacecolor="white", markeredgecolor=MUTED, label="Median")]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left - 0.012, 0.958),
                     bbox_transform=fig.transFigure, ncol=3, columnspacing=1.4)
    fit_legend(fig, leg)

    return save(fig, out_dir, "Fig 4. Contract term and delivery changes")


# ----------------------------------------------------------------------------
# Figure 5 — the joint seller/buyer movement, on one comparable scale
# ----------------------------------------------------------------------------

def symlog_axis(ax, which: str, linthresh: float, limit: float) -> None:
    (ax.set_xscale if which == "x" else ax.set_yscale)(
        "symlog", linthresh=linthresh, linscale=0.45, base=10)
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(FixedLocator([-100, -10, -1, 0, 1, 10, 100]))
    axis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{'\u2212' if v < 0 else ''}{abs(v):g}"))
    axis.set_minor_locator(FixedLocator([-1000, -0.1, 0.1, 1000]))
    (ax.set_xlim if which == "x" else ax.set_ylim)(-limit, limit)


def build_fig5(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    long = tables["long"].copy()
    long["dx"] = long.delta_seller_metric / 1e6              # $ million
    long["dy"] = long.delta_buyer_participation_slack / 1e6  # $ million

    fig = plt.figure(figsize=(PRINT_WIDTH_IN, 4.95))
    left, right, bottom, top = 0.093, 0.995, 0.072, 0.895
    col_gap, row_gap = 0.070, 0.105
    pw = (right - left - col_gap) / 2
    ph = (top - bottom - row_gap) / 2
    axes = [fig.add_axes((left + (i % 2) * (pw + col_gap),
                          bottom + (1 - i // 2) * (ph + row_gap), pw, ph))
            for i in range(4)]

    linthresh, limit = 0.1, 800.0
    # (x, y, ha, va) of the corner belonging to each (dx>0, dy>0) quadrant.
    corners = {(False, True): (0.030, 0.968, "left", "top"),
               (True, True): (0.970, 0.968, "right", "top"),
               (False, False): (0.030, 0.032, "left", "bottom"),
               (True, False): (0.970, 0.032, "right", "bottom")}

    for ai, (ax, channel) in enumerate(zip(axes, CHANNEL_ORDER)):
        sub = long[long.mutation_family_normalized.eq(channel)]
        ax.axhline(0, color=INK, linewidth=0.7, zorder=1)
        ax.axvline(0, color=INK, linewidth=0.7, zorder=1)
        for lv in SHIFT_LEVELS:
            sel = sub[np.isclose(sub.target_shift_abs, lv)]
            ax.scatter(sel.dx, sel.dy, s=5.0, c=SHIFT_COLOR[lv], alpha=0.55,
                       linewidths=0, zorder=2, rasterized=True)
        for lv in SHIFT_LEVELS:
            sel = sub[np.isclose(sub.target_shift_abs, lv)]
            ax.scatter(sel.dx.median(), sel.dy.median(), s=44, marker="D",
                       c=SHIFT_COLOR[lv], edgecolors=INK, linewidths=0.8, zorder=5)

        # Exact quadrant composition, written in the corner of the quadrant it
        # counts, one line per shift size in the same colours as the points.
        for (x_pos, y_pos), (cx, cy, ha, va) in corners.items():
            for k, lv in enumerate(SHIFT_LEVELS):
                sel = sub[np.isclose(sub.target_shift_abs, lv)]
                share = float((((sel.dx > 0) == x_pos) & ((sel.dy > 0) == y_pos)).mean())
                step = 0.075 * (k if va == "top" else (len(SHIFT_LEVELS) - 1 - k))
                ax.text(cx, cy - step if va == "top" else cy + step, f"{share:.0%}",
                        transform=ax.transAxes, ha=ha, va=va, fontsize=ANNOT_FS,
                        color=SHIFT_COLOR[lv], zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.8))

        symlog_axis(ax, "x", linthresh, limit)
        symlog_axis(ax, "y", linthresh, limit)
        ax.set_title(CHANNEL_WRAPPED[channel].replace(chr(10), " "),
                     fontsize=BASE_FS, color=INK, pad=4)
        ax.grid(True, which="major", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(which="minor", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ai % 2:
            ax.set_yticklabels([])
        if ai < 2:
            ax.set_xticklabels([])

    fig.text(0.012, bottom + ph + row_gap / 2, "Δ buyer participation slack (\\$ million)",
             rotation=90, ha="left", va="center", fontsize=BASE_FS)
    fig.text((left + right) / 2, 0.012, "Δ seller exposure index (\\$ million)",
             ha="center", va="bottom", fontsize=BASE_FS)
    # The leading entry carries no marker; it names what the colours encode
    # without a legend title, which would cost a second line.
    handles = [Line2D([], [], linestyle="none", label="Shift size:")]
    handles += [Line2D([], [], marker="o", linestyle="none", markersize=3.4,
                       markerfacecolor=SHIFT_COLOR[lv], markeredgecolor="none",
                       label=f"{SHIFT_NAME[lv]} ({lv:.2f})") for lv in SHIFT_LEVELS]
    handles.append(Line2D([], [], marker="D", linestyle="none", markersize=4.4,
                          markerfacecolor="white", markeredgecolor=INK,
                          markeredgewidth=0.8, label="median of each shift size"))
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.075, 1.115),
                   ncol=5, handletextpad=0.3, columnspacing=1.1)

    return save(fig, out_dir, "Fig 5. Seller exposure and buyer slack")


# ----------------------------------------------------------------------------

def save(fig, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=DPI)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  wrote {png.name}  (+ .pdf)")
    return png


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", default=str(here.parent))
    ap.add_argument("--out-dir", default=str(here / "redesign_output"))
    ap.add_argument("--also-copy-to", default=None,
                    help="second directory to copy the PNG and PDF files into")
    ap.add_argument("--verify", action="store_true",
                    help="re-derive the published reference numbers from the same tables")
    args = ap.parse_args()

    set_style()
    tables = load_tables(Path(args.code_root).expanduser().resolve())
    if args.verify:
        verify(tables)

    out_dir = Path(args.out_dir).expanduser().resolve()
    print(f"Writing redesigned figures to {out_dir}")
    build_fig2(tables, out_dir)
    build_fig3(tables, out_dir)
    build_fig4(tables, out_dir)
    build_fig5(tables, out_dir)

    if args.also_copy_to:
        dest = Path(args.also_copy_to).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        files = sorted(out_dir.glob("Fig *"))
        for p in files:
            shutil.copy2(p, dest / p.name)
        print(f"Copied {len(files)} files to {dest}")


if __name__ == "__main__":
    main()
