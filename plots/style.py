"""Shared labels, palette and axis styling for the figure scripts.

The three plotting scripts drew from the same palette and the same label
dictionaries, copied into each file. That was survivable with two models; it
stops being survivable the moment a model is added, because the figures then
disagree with each other about what colour a model is - which is exactly the
kind of inconsistency a reader reads as meaning.

Everything a figure needs to know about *which* models and targets exist is
still read off results/benchmark_scores.csv at run time. This module only says
how to draw them.
"""

import matplotlib.pyplot as plt


# ======================================================
# LABELS
# ======================================================

MODEL_LABELS = {
    "historical_20y": "Historical quantiles",
    "qar": "Quantile AR",
    "chronos2": "Chronos-2 (zero-shot)",
    "qar_nfci": "Quantile AR + NFCI",
    "chronos2_nfci": "Chronos-2 + NFCI"
}

TARGET_LABELS = {
    "cpi_yoy": "CPI inflation, year-on-year (%)",
    "ipi_growth": "Industrial production, monthly growth (%)",
    "ipi_growth_12m": "Industrial production, 12-month growth (%)",
    "gdp_growth": "Real GDP, quarterly growth (%)"
}

COUNTRY_LABELS = {
    "DEU": "Germany", "FRA": "France",
    "GBR": "United Kingdom", "USA": "United States"
}


def model_label(name):
    """Display name for a model, falling back to its key."""

    return MODEL_LABELS.get(name, name)


def target_label(name):
    """Display name for a target, falling back to its key."""

    return TARGET_LABELS.get(name, name)


# ======================================================
# PALETTE
# ======================================================
#
# Blue / orange / bluish-green, the first three of Okabe-Ito: separable under
# the common forms of colour vision deficiency, which a blue/green/orange
# triad picked by eye is not. The order is fixed so a model keeps its colour
# across every figure.

MODEL_COLORS = {
    "historical_20y": "#2a78d6",
    "qar": "#eb6834",
    "chronos2": "#0e8f6f",
    # A conditional model keeps the hue of the univariate model it nests, one
    # shade darker: the pair is meant to be read as a pair, and a fourth and
    # fifth unrelated hue would hide that.
    "qar_nfci": "#a83c12",
    "chronos2_nfci": "#08553f"
}

# Anything not in the table above, in order, so an unrecognised model is still
# drawn rather than raising in the middle of a figure.
FALLBACK_COLORS = ["#8a5fb0", "#b8912f", "#4a3aa7", "#a03d5f"]

DIFF_COLOR = "#4a3aa7"
REALISED = "#eb6834"

# One colour per country, so a country keeps its colour across the data
# figures the same way a model does across the evaluation figures.
COUNTRY_COLORS = {
    "DEU": "#2a78d6",
    "FRA": "#eb6834",
    "GBR": "#0e8f6f",
    "USA": "#b8912f"
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Sequential blue ramp: time runs light -> dark.
TIME_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def model_color(name, used=None):
    """Colour for a model. Unknown names take the next unused fallback."""

    if name in MODEL_COLORS:
        return MODEL_COLORS[name]

    taken = set(used or [])
    spare = [c for c in FALLBACK_COLORS if c not in taken]

    return spare[0] if spare else INK_SECONDARY


def model_colors(names):
    """A colour per model, stable regardless of the order names arrive in."""

    out = {}

    for name in names:
        out[name] = model_color(name, used=out.values())

    return out


# ======================================================
# AXES
# ======================================================

def apply_rcparams(tick_size=9, axes_title_size=11, axes_label_size=10):
    """The shared figure style. Sizes vary by figure, so they are arguments."""

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "axes.titlesize": axes_title_size,
        "axes.labelsize": axes_label_size
    })


def style_axes(ax, grid_width=0.8):
    """Horizontal gridlines, no top/right spines, muted baselines."""

    ax.grid(axis="y", color=GRIDLINE, linewidth=grid_width)
    ax.set_axisbelow(True)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
