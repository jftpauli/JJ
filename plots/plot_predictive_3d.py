"""Predictive distributions over time, in 3D, with the realisation overlaid.

One ridge per forecast origin: the h = 1 predictive density, drawn in the
(value, density) plane and stacked along time. The realised value is drawn as a
line on the floor of the box, so you can see whether each month's outcome landed
in the fat part of that month's distribution or out in a tail.

This is the picture the scalar scores compress. A CRPS of 0.18 does not tell you
*where* a model was wrong; a ridge that stayed narrow and centred on 2% through
2021 while the realisation walked up to 9% does.

Densities are reconstructed from the stored quantiles: between adjacent levels
the density is the slope (tau_{j+1} - tau_j) / (Q_{j+1} - Q_j), evaluated at the
midpoint and interpolated onto a common grid. That is a coarse reconstruction -
it is piecewise-constant before smoothing, and the 1%/99% grid cannot show what
happens further out - so read the ridges for shape and location, not for fine
detail in the tails.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)

# Paths are anchored to the repository root rather than the working directory,
# so the script runs the same from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_FOLDER = os.path.join(ROOT, "results")
FIGURE_FOLDER = os.path.join(ROOT, "figures")

# What to draw. One country at a time - the ridges are unreadable if several
# countries are stacked into the same box.
TARGET = "cpi_yoy"
COUNTRY = "DEU"
HORIZON = 1

START = "2016-01-01"
END = "2025-12-01"
EVERY = 3               # keep every n-th origin, so the ridges stay separable

MODELS = ["historical_20y", "qar"]

MODEL_LABELS = {
    "historical_20y": "Historical quantiles",
    "qar": "Quantile AR"
}

TARGET_LABELS = {
    "cpi_yoy": "CPI inflation, year-on-year (%)",
    "ipi_growth": "Industrial production, monthly growth (%)"
}

COUNTRY_LABELS = {
    "DEU": "Germany", "FRA": "France",
    "GBR": "United Kingdom", "USA": "United States"
}

# Sequential blue ramp: time runs light -> dark.
TIME_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

REALISED = "#eb6834"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

N_GRID = 220

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})


def ramp_color(fraction):
    """Sample the sequential ramp at a position in [0, 1]."""

    index = int(round(fraction * (len(TIME_RAMP) - 1)))

    return TIME_RAMP[min(max(index, 0), len(TIME_RAMP) - 1)]


def density_from_quantiles(values, levels, grid, smoothing=2.5):
    """Reconstruct a density on `grid` by differentiating the CDF.

    The obvious route - density = dtau / dq between adjacent quantile levels -
    is numerically unusable here. Where two fitted quantiles nearly coincide,
    dq approaches zero and the density explodes; on this data that produced
    spikes two orders of magnitude above the rest of the distribution, which
    say nothing about the forecast and swamp the plot.

    Instead the quantile function is inverted to a CDF on the grid and
    differentiated, then lightly smoothed. Interpolating the CDF is stable
    because it is monotone and bounded, so a near-tie in the quantiles produces
    a locally steep but finite slope rather than a pole.
    """

    order = np.argsort(values)
    cdf = np.interp(
        grid, values[order], levels[order],
        left=levels[0], right=levels[-1]
    )

    cdf = gaussian_filter1d(cdf, smoothing, mode="nearest")
    density = np.gradient(cdf, grid)

    return np.clip(density, 0.0, None)


def load_slice(model):
    """Predictive quantiles and realisations for the configured slice."""

    scores = pd.read_csv(
        os.path.join(RESULTS_FOLDER, "benchmark_scores.csv"),
        parse_dates=["origin"]
    )

    bundle = np.load(os.path.join(RESULTS_FOLDER, "predictive_quantiles.npz"))
    predictions = bundle["predictions"]
    levels = bundle["quantile_levels"]

    if len(scores) != len(predictions):
        raise ValueError(
            "score rows and prediction rows are out of step - re-run "
            "run_benchmarks.py"
        )

    mask = (
        (scores.target == TARGET)
        & (scores.country == COUNTRY)
        & (scores.model == model)
        & (scores.horizon == HORIZON)
        & (scores.origin >= pd.Timestamp(START))
        & (scores.origin <= pd.Timestamp(END))
    ).to_numpy()

    subset = scores[mask].reset_index(drop=True)
    preds = predictions[mask]

    keep = np.arange(0, len(subset), EVERY)

    return subset.iloc[keep].reset_index(drop=True), preds[keep], levels


def draw_panel(ax, subset, preds, levels, title, grid, z_max, show_zlabel):

    n = len(subset)

    densities = np.array([
        density_from_quantiles(preds[i], levels, grid) for i in range(n)
    ])

    # Ridges as filled polygons in the (value, density) plane, placed along time.
    polygons = []
    colors = []

    for i in range(n):
        vertices = [(grid[0], 0.0)]
        vertices += list(zip(grid, densities[i]))
        vertices += [(grid[-1], 0.0)]

        polygons.append(vertices)
        colors.append(ramp_color(i / max(n - 1, 1)))

    collection = PolyCollection(
        polygons,
        facecolors=[(*plt.matplotlib.colors.to_rgb(c), 0.72) for c in colors],
        edgecolors=[(*plt.matplotlib.colors.to_rgb(c), 1.0) for c in colors],
        linewidths=0.6
    )

    ax.add_collection3d(collection, zs=np.arange(n), zdir="x")

    # The realisation, on the floor of the box.
    ax.plot(
        np.arange(n),
        subset.actual.to_numpy(),
        zs=0.0,
        zdir="z",
        color=REALISED,
        linewidth=2.0,
        label="realised"
    )

    ax.scatter(
        np.arange(n),
        subset.actual.to_numpy(),
        np.zeros(n),
        color=REALISED,
        s=9,
        depthshade=False
    )

    ax.set_xlim(0, n - 1)
    ax.set_ylim(grid[0], grid[-1])
    ax.set_zlim(0, z_max)

    # Year ticks along the time axis.
    dates = pd.DatetimeIndex(subset.origin)
    tick_positions, tick_labels = [], []

    for i, date in enumerate(dates):
        if date.month == 1 and date.year % 2 == 0:
            tick_positions.append(i)
            tick_labels.append(str(date.year))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("forecast origin", labelpad=8)
    ax.set_ylabel(TARGET_LABELS[TARGET], labelpad=8)
    # Both panels share the density scale, so it is labelled once.
    if show_zlabel:
        ax.set_zlabel("density", labelpad=2)

    ax.set_title(title, color=INK_PRIMARY, fontweight="bold", pad=0)

    ax.view_init(elev=24, azim=-58)

    ax.xaxis.pane.set_facecolor(SURFACE)
    ax.yaxis.pane.set_facecolor(SURFACE)
    ax.zaxis.pane.set_facecolor(SURFACE)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("#e1e0d9")

    ax.grid(False)
    ax.tick_params(colors=INK_MUTED)


def main():

    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    fig = plt.figure(figsize=(15, 6.6))

    slices = {}

    for model in MODELS:

        subset, preds, levels = load_slice(model)

        if len(subset) == 0:
            raise ValueError(
                f"no forecasts for {TARGET}/{COUNTRY}/{model} in "
                f"{START}..{END} - check the configuration at the top"
            )

        slices[model] = (subset, preds, levels)

    # One value grid and one density scale for every panel. Without this the
    # sharper model is drawn on a taller z-axis and the two look equally
    # confident, which is the opposite of what the plot is meant to show.
    low = min(min(p[:, 0].min(), s.actual.min()) for s, p, _ in slices.values())
    high = max(max(p[:, -1].max(), s.actual.max()) for s, p, _ in slices.values())
    pad = 0.08 * (high - low)
    grid = np.linspace(low - pad, high + pad, N_GRID)

    z_max = 1.05 * max(
        max(
            density_from_quantiles(preds[i], levels, grid).max()
            for i in range(len(subset))
        )
        for subset, preds, levels in slices.values()
    )

    for position, model in enumerate(MODELS, start=1):

        subset, preds, levels = slices[model]

        ax = fig.add_subplot(1, len(MODELS), position, projection="3d")

        draw_panel(
            ax, subset, preds, levels, MODEL_LABELS[model], grid, z_max,
            show_zlabel=(position == 1)
        )

    fig.suptitle(
        f"h = {HORIZON} predictive distributions over time - "
        f"{COUNTRY_LABELS[COUNTRY]}, {TARGET_LABELS[TARGET]}",
        x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK_PRIMARY
    )

    fig.text(
        0.02, 0.02,
        "Each ridge is one month's predictive density (reconstructed from the "
        "1-99% quantile grid); ridges run light to dark with time. "
        f"The orange line on the floor is the realisation. Every {EVERY}rd "
        "origin shown.",
        fontsize=8.5, color=INK_MUTED
    )

    # tight_layout cannot fit 3D axis decorations; set margins directly
    # so the right panel's z-label is not clipped.
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.06, top=0.92, wspace=0.10)

    path = os.path.join(
        FIGURE_FOLDER,
        f"predictive_3d_{TARGET}_{COUNTRY}.png"
    )

    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", path)


if __name__ == "__main__":
    main()
