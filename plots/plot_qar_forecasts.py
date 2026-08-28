"""Rolling-origin Quantile AR forecasts against the realised series.

One panel per country, one figure per horizon. Each panel shows three things
dated at the *target* period, not the origin: the realised year-on-year
inflation rate, the QAR median forecast, and the 90% prediction interval
(q0.05 to q0.95).

Plotting at the target date is the only alignment that lets a reader check the
forecast. Dated at the origin, the h = 12 band would sit a year to the left of
the observation it is a statement about, and the eye would compare it to the
wrong number.

The two horizons get separate figures rather than a shared axis. Their bands
differ in width by roughly a factor of three, so overlaying them makes the
h = 1 band a hairline; and the h = 12 forecast for a given target date is made
from a different information set, so the two are not alternative readings of
the same quantity that a reader should be scanning between.

Coverage - the share of realised values inside the band - is annotated per
panel. A calibrated 90% interval covers 90% of them; the rolling-origin
forecasts are out-of-sample, so the number is a real diagnostic rather than an
in-sample tautology.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import (  # noqa: E402
    BASELINE,
    COUNTRY_LABELS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    apply_rcparams,
    model_color,
    model_label,
    style_axes,
    target_label,
)

RESULTS = os.path.join(ROOT, "results", "quantile_forecasts.csv")
PANEL = os.path.join(ROOT, "data", "OECD_cpi_yoy_monthly_panel.csv")
FIGURE_FOLDER = os.path.join(ROOT, "figures")

TARGET = "cpi_yoy"
LOWER, UPPER, MEDIAN = "q0.05", "q0.95", "q0.5"

SOURCE = (
    "Source: OECD CPI, year-on-year. Quantile AR fit by rolling origin from "
    "2005-01, re-estimated at every origin on the data available then; p = 13 "
    "at h = 1, p = 1 at h = 12. Band is the 0.05-0.95 predictive interval, "
    "plotted at the target date. Coverage is the out-of-sample share of "
    "realised values inside the band."
)

QAR = model_color("qar")

apply_rcparams(tick_size=9)


def load():
    """Forecasts re-dated from the origin to the period being forecast."""

    forecasts = pd.read_csv(RESULTS, parse_dates=["origin"])

    forecasts["target_date"] = forecasts.apply(
        lambda r: r["origin"] + pd.DateOffset(months=int(r["horizon"])), axis=1
    )

    realised = pd.read_csv(PANEL, index_col="TIME_PERIOD", parse_dates=True)

    return forecasts, realised


def plot(forecasts, realised, h):

    frame = forecasts[forecasts["horizon"] == h]
    countries = list(realised.columns)

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6.4), sharex=True, sharey=True, squeeze=False
    )

    for ax, country in zip(axes.ravel(), countries):

        panel = frame[frame["country"] == country].set_index("target_date")
        panel = panel.sort_index()

        ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)

        ax.fill_between(
            panel.index, panel[LOWER], panel[UPPER],
            color=QAR, alpha=0.28, linewidth=0, zorder=2,
            label="90% prediction interval"
        )

        ax.plot(
            panel.index, panel[MEDIAN],
            color=QAR, linewidth=1.4, zorder=3,
            label=f"{model_label('qar')}, median"
        )

        # The realised series over the same window, drawn last and in ink: it
        # is the thing every other mark is a claim about.
        truth = realised[country].reindex(panel.index)

        ax.plot(
            panel.index, truth.to_numpy(),
            color=INK_PRIMARY, linewidth=1.1, zorder=4,
            label="Realised"
        )

        inside = ((truth >= panel[LOWER]) & (truth <= panel[UPPER]))
        covered = inside[truth.notna()].mean()

        ax.set_title(
            COUNTRY_LABELS.get(country, country),
            loc="left", color=INK_PRIMARY, fontweight="bold", pad=6
        )

        ax.annotate(
            f"coverage {covered:.0%}",
            xy=(0.015, 0.94), xycoords="axes fraction",
            ha="left", va="top", fontsize=8.5, color=INK_MUTED
        )

        style_axes(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("% change on a year earlier")

    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Realised first: the legend should read in the order the eye needs the
    # marks, not the order they were drawn.
    order = [labels.index(x) for x in
             ["Realised", f"{model_label('qar')}, median",
              "90% prediction interval"]]

    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper right", bbox_to_anchor=(0.995, 0.995),
        ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY
    )

    fig.suptitle(
        f"{target_label(TARGET)} - Quantile AR, {h}-month horizon",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK_PRIMARY
    )

    fig.text(0.01, 0.005, SOURCE, fontsize=8, color=INK_MUTED)

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    path = os.path.join(FIGURE_FOLDER, f"qar_forecasts_h{h}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():

    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    forecasts, realised = load()

    for h in sorted(forecasts["horizon"].unique()):
        plot(forecasts, realised, int(h))


if __name__ == "__main__":
    main()
