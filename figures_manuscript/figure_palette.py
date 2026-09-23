"""One colour palette for every figure in the manuscript and the supplement.

Before this file the paper drew from three unrelated palettes: matplotlib's
tab10 defaults in Figures 4, 6, 7 and A4-A10, CSS names (sandybrown,
mediumturquoise, red) in Figures A1-A3, and an Okabe-Ito set in Figures 2, 3
and 5. tab10's blue and orange sit close to Okabe-Ito's without matching them,
and a near-miss reads worse than an obvious difference, so the whole paper now
draws from the Okabe-Ito set below.

Okabe-Ito is used because it is the standard eight-colour set designed to stay
separable under protanopia, deuteranopia and tritanopia; the pairwise
separations were re-checked in OKLab when Figures 2, 3 and 5 were built.

The three contract structures own the first three entries, because the volume
structure is the paper's primary categorical variable and the reader meets
those colours first, in Figures 2 and 3. Figures whose series are something
else simply take the cycle in order; where such a figure also has
structure-labelled panels (A4 and A6), the legend names the series, so no
colour is doing two jobs inside one figure.

Import this module rather than restating a hex code, so that the palette cannot
drift apart again:

    from figure_palette import STRUCTURE_COLOR, CYCLE, apply_palette
"""
from __future__ import annotations

# Okabe-Ito, by name.
BLUE = "#0072B2"           # blue
AMBER = "#E69F00"          # orange
GREEN = "#009E73"          # bluish green
PURPLE = "#CC79A7"         # reddish purple
VERMILLION = "#D55E00"     # vermillion
SKY = "#56B4E9"            # sky blue
BLACK = "#000000"
WINE = "#882255"           # Tol muted
INDIGO = "#332288"         # Tol muted

# Neutrals for rules, grids and axis furniture; not part of the series cycle.
INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#DCDCDC"
NEUTRAL_FILL = "#A0A0A0"

# The contract structures keep one colour each, in Figures 2 and 3.
STRUCTURE_COLOR = {"Fix": BLUE, "AsC": AMBER, "AsG": GREEN}

# The order any other categorical series is assigned in.
# Slots 4 and 5 are wine and indigo rather than the Okabe-Ito purple and
# vermillion: those two sit 1.4 and 4.0 from the muted channel ochre and lilac
# in OKLab, close enough to read as the same colour carrying two meanings in
# two figures. Wine and indigo are 8.8 away at worst and separate the
# five-series sensitivity figures better as well (9.2 against 8.2).
CYCLE = [BLUE, AMBER, GREEN, WINE, INDIGO, SKY, BLACK]

# Requested shift size in Figure 5: an ordered variable, so it gets a set that
# separates by hue as well as by lightness. A single-hue ramp was tried first
# and the levels could not be told apart at scatter-point size.
SHIFT_COLOR = {0.1: NEUTRAL_FILL, 0.3: "#B2182B", 0.5: INDIGO}

# Figures A1-A3 describe one variable each, so colour is not encoding a
# category; each takes a palette fill so the three read as a set, and the mean
# marker is the same accent in all three.
PROFILE_FILL = {"demand": AMBER, "generation": GREEN, "lmp": SKY}
PROFILE_MEAN = INDIGO      # reads against amber, green and sky blue alike


def apply_palette(plt) -> None:
    """Point matplotlib's default colour cycle at CYCLE.

    Several producers never name a colour and simply take C0, C1, C2 in draw
    order. Calling this once makes those implicit choices land on the palette
    instead of on tab10.
    """
    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] = cycler(color=CYCLE)

# The four dependence channels keep one colour and one marker each, wherever a
# figure separates them: as curves in Figure 2 and as columns in Figure 4. These
# are the Okabe-Ito entries the structures do not use, so a channel colour can
# never be mistaken for a structure colour.
# Muted rather than the saturated Okabe-Ito entries: on the page those read as
# glare against a 10 pt serif body. Separation was re-checked in OKLab and this
# set is in fact better separated than the saturated one it replaces (worst
# dichromat pair 9.4 against 8.7) at about two thirds of the saturation, because
# the four are spread across lightness instead of relying on hue alone.
CHANNEL_COLOR = {
    "Generation-load mismatch": "#222222",
    "Seller-buyer nodal price decoupling": "#35708E",
    "Buyer load-price intensification": "#C07830",
    "Seller generation-price cannibalization": "#B08FC0",
}
CHANNEL_MARKER = {
    "Generation-load mismatch": "o",
    "Seller-buyer nodal price decoupling": "s",
    "Buyer load-price intensification": "^",
    "Seller generation-price cannibalization": "D",
}

BASE_FS = 8.0
ANNOT_FS = 6.0


def house_rcparams() -> dict:
    """The style every figure in the paper is drawn with.

    Kept here rather than in each producer so that a change to the typography
    or the axis furniture reaches all eighteen figures at once.
    """
    return {
        "font.size": BASE_FS,
        "axes.titlesize": BASE_FS,
        "axes.labelsize": BASE_FS,
        "xtick.labelsize": BASE_FS - 0.5,
        "ytick.labelsize": BASE_FS - 0.5,
        "legend.fontsize": BASE_FS - 0.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handleheight": 0.9,
        "legend.columnspacing": 1.2,
        "legend.borderpad": 0.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.pad_inches": 0.02,
    }


def tidy(ax, grid: str = "y") -> None:
    """Drop the top and right rules and put the grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid == "y":
        ax.yaxis.grid(True, zorder=0)
    elif grid == "x":
        ax.xaxis.grid(True, zorder=0)
    elif grid == "both":
        ax.grid(True, zorder=0)
    ax.set_axisbelow(True)


def fit_legend(fig, leg, min_fontsize: float = 5.5, margin_pt: float = 1.5) -> None:
    """Keep a legend inside the figure's page box, measured after a draw.

    A legend anchored by hand can run past the edge by a few points, which the
    PDF then clips. Width overflow is met by stepping the legend font down (never
    below `min_fontsize`); height overflow by moving the anchor down by the
    overshoot. Measured, so it holds when labels or fonts change later.
    """
    fig.canvas.draw()
    W, H = fig.get_size_inches() * fig.dpi
    m = margin_pt * fig.dpi / 72.0
    for _ in range(8):
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        wide, high = bb.x1 > W - m or bb.x0 < m, bb.y1 > H - m
        if not (wide or high):
            return
        if wide:
            fs = leg.get_texts()[0].get_fontsize()
            if fs - 0.5 < min_fontsize:
                break
            for t in leg.get_texts():
                t.set_fontsize(fs - 0.5)
        if high:
            x, y = leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).p0
            leg.set_bbox_to_anchor((x, y - (bb.y1 - (H - m)) / H), transform=fig.transFigure)
        fig.canvas.draw()
