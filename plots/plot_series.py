"""The raw data figure: the forecasting target, one panel per country.

Growth, not the index. The industrial production index is non-stationary - it
trends, it has been re-based, and its level is not comparable across countries
(different base periods), so an index chart shows mostly the trend and the
choice of base year. The monthly log difference is what the models are actually
fitted to, and it is the series whose behaviour a reader needs to see: the
volatility differs by an order of magnitude across countries, and the shocks
(2008-09, 2020) are the observations that dominate any average score.

Small multiples rather than four lines on one axis. Monthly growth is noisy
enough that overlaying four countries produces a hairball in which nothing but
the COVID spike is legible.

The y-axis is clipped. March-June 2020 reaches -25pp, twelve standard
deviations out; on a full-range axis every other month is a flat band and the
figure says nothing except that 2020 happened. Clipped months are drawn as
markers on the frame and named in the panel, so nothing is hidden.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Paths are anchored to the repository root rather than the working directory,
# so the script runs the same from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import (  # noqa: E402
    BASELINE,
    COUNTRY_COLORS,
    COUNTRY_LABELS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    apply_rcparams,
    style_axes,
    target_label,
)

DATA_FOLDER = os.path.join(ROOT, "data")
FIGURE_FOLDER = os.path.join(ROOT, "figures")

PANEL = "OECD_ipi_growth_monthly_panel.csv"
TARGET = "ipi_growth"
FIGURE = "ipi_growth.png"

SOURCE = (
    "Source: OECD, DSD_STES@DF_INDSERV (PRVM). Industry except construction, "
    "calendar and seasonally adjusted. Monthly log difference, 100 x dlog. "
    "1990-2025. Axis clipped at +/-8pp; months beyond it are marked on the "
    "frame with their value."
)

# Wide enough to hold everything outside the pandemic quarter, tight enough
# that ordinary month-to-month variation is still readable.
YLIM = 8.0

apply_rcparams(tick_size=9)


def load():

    return pd.read_csv(
        os.path.join(DATA_FOLDER, PANEL),
        index_col="TIME_PERIOD",
        parse_dates=["TIME_PERIOD"]
    )


def plot(panel):

    countries = list(panel.columns)

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6.4), sharex=True, sharey=True, squeeze=False
    )

    for ax, country in zip(axes.ravel(), countries):

        series = panel[country].dropna()
        color = COUNTRY_COLORS.get(country, INK_SECONDARY)

        ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)

        ax.plot(
            series.index, series.to_numpy(),
            color=color,
            linewidth=0.7,
            zorder=2
        )

        ax.set_ylim(-YLIM, YLIM)

        # Clipped months, pinned to the frame so the reader sees that the line
        # leaves the axis rather than that the series stops.
        outside = series[series.abs() > YLIM]

        for date, value in outside.items():
            ax.plot(
                date, YLIM if value > 0 else -YLIM,
                marker="^" if value > 0 else "v",
                markersize=4.5, color=color, clip_on=False, zorder=4
            )

        if len(outside):
            trough = outside.idxmin()
            ax.annotate(
                f"{trough:%b %Y}: {outside.min():.1f}",
                xy=(0.985, 0.06), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5, color=INK_MUTED
            )

        ax.set_title(
            COUNTRY_LABELS.get(country, country),
            loc="left", color=INK_PRIMARY, fontweight="bold", pad=6
        )

        # The standard deviation is the number that matters for reading the
        # scores: CRPS is in the units of the target, so a country with twice
        # the volatility carries twice the score at equal skill.
        ax.annotate(
            f"sd = {series.std():.2f} pp",
            xy=(0.985, 0.94), xycoords="axes fraction",
            ha="right", va="top", fontsize=8.5, color=INK_MUTED
        )

        style_axes(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("% change on previous month")

    fig.suptitle(
        target_label(TARGET),
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK_PRIMARY
    )

    fig.text(0.01, 0.005, SOURCE, fontsize=8, color=INK_MUTED)

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    path = os.path.join(FIGURE_FOLDER, FIGURE)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():

    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    plot(load())


if __name__ == "__main__":
    main()
