"""Rolling-origin forecasts against the realised series, one figure per model.

One panel per country, one figure per model and horizon. Each panel shows three
things dated at the *target* period, not the origin: the realised year-on-year
inflation rate, the median forecast, and the 90% prediction interval (q0.05 to
q0.95).

Plotting at the target date is the only alignment that lets a reader check the
forecast. Dated at the origin, the h = 12 band would sit a year to the left of
the observation it is a statement about, and the eye would compare it to the
wrong number.

The horizons get separate figures rather than a shared axis. Their bands differ
in width by roughly a factor of three, so overlaying them makes the h = 1 band a
hairline; and the h = 12 forecast for a given target date is made from a
different information set, so the two are not alternative readings of the same
quantity that a reader should be scanning between.

Coverage - the share of realised values inside the band - is annotated per
panel. A calibrated 90% interval covers 90% of them; the rolling-origin
forecasts are out-of-sample, so the number is a real diagnostic rather than an
in-sample tautology. Read it against the width of the band, not on its own: the
climatological baseline can hit 90% simply by being wide enough to be useless.

Every model in results/ gets the same treatment, so the figures are comparable
by construction - same axes, same band, same coverage definition.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]

from config import COUNTRIES, DATA_PATHS, FORECAST_START_DATE, RESULTS_PATH  # noqa: E402
from models.qar import LAGS  # noqa: E402

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

PANEL = DATA_PATHS["cpi"]
FIGURE_FOLDER = os.path.join(ROOT, "figures")

TARGET = "cpi_yoy"
LOWER, UPPER, MEDIAN = "q0.05", "q0.95", "q0.5"

# What the reader has to know to judge each band. The rolling-origin scheme is
# common to all three; what differs is the information each one conditions on.
NOTES = {
    "historical": ("Empirical quantiles of every observation up to the origin, "
                   "so the forecast conditions on nothing and is identical at "
                   "both horizons"),
    "qar": ("Quantile regression re-estimated at every origin on the data "
            "available then; "
            + ", ".join(f"p = {p} at h = {h}" for h, p in sorted(LAGS.items()))),
    "chronos2": ("Pretrained Chronos-2 applied zero-shot, no fitting on this "
                 "series; its training window overlaps the evaluation period, "
                 "so leakage is not controlled"),
}

apply_rcparams(tick_size=9)


def load(model):
    """Forecasts re-dated from the origin to the period being forecast."""

    forecasts = pd.read_csv(
        RESULTS_PATH / f"{model}_forecasts.csv", parse_dates=["origin"]
    )

    forecasts["target_date"] = forecasts.apply(
        lambda r: r["origin"] + pd.DateOffset(months=int(r["horizon"])), axis=1
    )

    return forecasts


def plot(model, forecasts, realised, h):

    frame = forecasts[forecasts["horizon"] == h]
    color = model_color(model)

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6.4), sharex=True, sharey=True, squeeze=False
    )

    for ax, country in zip(axes.ravel(), COUNTRIES):

        panel = frame[frame["country"] == country].set_index("target_date")
        panel = panel.sort_index()

        ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)

        ax.fill_between(
            panel.index, panel[LOWER], panel[UPPER],
            color=color, alpha=0.28, linewidth=0, zorder=2,
            label="90% prediction interval"
        )

        ax.plot(
            panel.index, panel[MEDIAN],
            color=color, linewidth=1.4, zorder=3,
            label=f"{model_label(model)}, median"
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

        # Median band width alongside coverage: the two only mean something
        # together, since any interval can be made to cover by widening it.
        width = (panel[UPPER] - panel[LOWER]).median()

        ax.set_title(
            COUNTRY_LABELS.get(country, country),
            loc="left", color=INK_PRIMARY, fontweight="bold", pad=6
        )

        ax.annotate(
            f"coverage {covered:.0%}   width {width:.1f}pp",
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
             ["Realised", f"{model_label(model)}, median",
              "90% prediction interval"]]

    fig.suptitle(
        f"{target_label(TARGET)} - {model_label(model)}, {h}-month horizon",
        x=0.01, y=0.995, ha="left", va="top",
        fontsize=13, fontweight="bold", color=INK_PRIMARY
    )

    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper left", bbox_to_anchor=(0.01, 0.945),
        ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY
    )

    fig.text(
        0.01, 0.005,
        f"Source: OECD CPI, year-on-year. Rolling origin from "
        f"{FORECAST_START_DATE[:7]}. {NOTES.get(model, '')}. Band is the "
        "0.05-0.95 predictive interval, plotted at the target date. Coverage "
        "and width are out-of-sample.",
        fontsize=8, color=INK_MUTED, wrap=True
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.90])

    path = os.path.join(FIGURE_FOLDER, f"{model}_forecasts_h{h}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():

    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    realised = pd.read_csv(PANEL, index_col="TIME_PERIOD", parse_dates=True)

    for path in sorted(RESULTS_PATH.glob("*_forecasts.csv")):

        model = path.name.removesuffix("_forecasts.csv")
        forecasts = load(model)

        for h in sorted(forecasts["horizon"].unique()):
            plot(model, forecasts, realised, int(h))


if __name__ == "__main__":
    main()
