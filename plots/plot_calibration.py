"""Calibration diagnostics for the benchmark predictive distributions.

Two complementary views, both at h = 1 only. That restriction is deliberate:
one-step forecasts made every month do not overlap, so the diagnostics below
are computed on (approximately) independent observations. Pooling h = 1..12
would stack twelve overlapping forecasts of the same month on top of each
other, and the reference bands would be badly overoptimistic.

PIT histogram - *probabilistic* calibration.
    The probability integral transform u_t = F_t(y_t). If the predictive
    distributions are calibrated, the u_t are uniform. The shape names the
    failure: a hump in the middle means the forecasts are too wide, a U shape
    means too narrow (overconfident), a tilt means bias.

Marginal calibration - Gneiting, Balabdaoui & Raftery (2007).
    The difference between the *average* predictive CDF and the empirical CDF
    of what actually happened,

        Delta(x) = (1/n) sum_t F_t(x)  -  Ghat(x)

    plotted against x. Zero everywhere means the forecasts get the
    unconditional distribution right. It is a weaker requirement than
    probabilistic calibration and catches a different failure: a model can be
    marginally calibrated while conditioning on nothing (the climatological
    baseline is the extreme case), so read the two plots together, never
    either alone.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths are anchored to the repository root rather than the working directory,
# so the script runs the same from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import (  # noqa: E402
    BASELINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    apply_rcparams,
    model_colors,
    model_label,
    style_axes,
    target_label,
)

RESULTS_FOLDER = os.path.join(ROOT, "results")
FIGURE_FOLDER = os.path.join(ROOT, "figures")

HORIZON = 1

N_BINS = 20

apply_rcparams(tick_size=9)


def load():

    scores = pd.read_csv(
        os.path.join(RESULTS_FOLDER, "benchmark_scores.csv"),
        parse_dates=["origin"]
    )

    bundle = np.load(
        os.path.join(RESULTS_FOLDER, "predictive_quantiles.npz")
    )

    return scores, bundle["predictions"], bundle["quantile_levels"]


def models_by_target(scores):
    """Which models were scored on each target, in a fixed display order.

    Read off the results rather than hardcoded: the model set is per-target
    (see TARGETS in run_benchmarks.py), so a target that does not run the
    foundation model must not get an empty panel drawn for it.

    Ordered climatology first, then the estimated benchmark, then anything
    else, so the reference a reader compares against is always the left panel.
    """

    priority = {"historical_20y": 0, "qar": 1}

    return {
        target: sorted(
            group.model.unique(),
            key=lambda name: (priority.get(name, 2), name)
        )
        for target, group in scores.groupby("target")
    }


# ======================================================
# PIT HISTOGRAM
# ======================================================

def pit_histogram(pit_values, levels):
    """Bin PIT values, sending clipped values to the outer bins.

    np.interp clamps, so a realisation below the lowest predictive quantile
    comes back as exactly levels[0]. That is not a PIT of 0.01 - it means the
    realisation fell somewhere in [0, 0.01]. Assigning those to the first bin
    (and the mirror case to the last) is the correct reading, not a fudge.
    """

    u = np.asarray(pit_values, dtype=float).copy()

    below = u <= levels[0]
    above = u >= levels[-1]

    # Nudge inside the outermost bins so np.histogram places them correctly.
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    u[below] = edges[0] + 0.5 * (edges[1] - edges[0])
    u[above] = edges[-2] + 0.5 * (edges[-1] - edges[-2])

    counts, _ = np.histogram(u, bins=edges)

    return counts / counts.sum(), edges, below.sum(), above.sum()


def plot_pit(scores, levels):

    by_target = models_by_target(scores)
    targets = sorted(by_target)
    colors = model_colors(sorted({m for ms in by_target.values() for m in ms}))

    # The grid is as wide as the busiest target. Targets running fewer models
    # leave empty cells, which are removed rather than left as blank boxes a
    # reader would take for a missing result.
    n_columns = max(len(ms) for ms in by_target.values())

    fig, axes = plt.subplots(
        len(targets), n_columns,
        figsize=(5.5 * n_columns, 3.5 * len(targets)),
        sharey="row",
        squeeze=False
    )

    for row, target in enumerate(targets):

        models = by_target[target]

        for ax in axes[row, len(models):]:
            ax.remove()

        for col, model in enumerate(models):

            ax = axes[row, col]

            subset = scores[
                (scores.target == target)
                & (scores.model == model)
                & (scores.horizon == HORIZON)
            ]

            share, edges, n_below, n_above = pit_histogram(
                subset.pit.to_numpy(), levels
            )

            ax.bar(
                edges[:-1],
                share,
                width=np.diff(edges),
                align="edge",
                color=colors[model],
                edgecolor=SURFACE,
                linewidth=1.0
            )

            # Uniform reference, plus a pointwise 95% band from the binomial
            # sampling distribution of a bin share.
            n = len(subset)
            expected = 1.0 / N_BINS
            half_band = 1.96 * np.sqrt(expected * (1 - expected) / n)

            ax.axhline(expected, color=INK_SECONDARY, linewidth=1.2, zorder=3)
            ax.axhspan(
                expected - half_band, expected + half_band,
                color=INK_SECONDARY, alpha=0.12, zorder=1
            )

            ax.set_title(
                f"{target_label(target)}\n{model_label(model)}",
                loc="left", color=INK_PRIMARY, fontweight="bold", pad=8
            )

            ax.set_xlim(0, 1)
            ax.set_xlabel("PIT value")

            if col == 0:
                ax.set_ylabel("share of forecasts")

            outside = (n_below + n_above) / n
            ax.annotate(
                f"n = {n:,}\n{outside:.1%} outside the 1-99% range",
                xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", va="top", fontsize=8.5, color=INK_SECONDARY
            )

            style_axes(ax)

    fig.suptitle(
        f"PIT histograms, h = {HORIZON}  -  flat is calibrated, "
        "U-shaped is overconfident",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK_PRIMARY
    )

    fig.text(
        0.01, 0.005,
        "Grey band: pointwise 95% interval for a bin share under uniformity. "
        "One-step forecasts, so the pooled observations do not overlap; "
        "countries are pooled, which the band does not adjust for.",
        fontsize=8, color=INK_MUTED
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    path = os.path.join(FIGURE_FOLDER, "pit_histograms.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


# ======================================================
# MARGINAL CALIBRATION
# ======================================================

def marginal_calibration(predictions, levels, actuals, grid):
    """Delta(x) = mean_t F_t(x) - Ghat(x) on the given grid."""

    # F_t(x) by interpolating each predictive quantile function. np.interp
    # clamps outside the grid, so F_t is pinned to [levels[0], levels[-1]];
    # with a 1%-99% grid that bounds the artefact at 0.01.
    mean_cdf = np.array([
        np.mean([
            np.interp(x, predictions[i], levels)
            for i in range(len(predictions))
        ])
        for x in grid
    ])

    empirical_cdf = np.array([
        np.mean(actuals <= x) for x in grid
    ])

    return mean_cdf - empirical_cdf


def plot_marginal_calibration(scores, predictions, levels):

    by_target = models_by_target(scores)
    targets = sorted(by_target)
    colors = model_colors(sorted({m for ms in by_target.values() for m in ms}))

    # Delta is a probability difference in both panels, so a shared y-axis is
    # meaningful here and stops the better-calibrated target from being read
    # as equally bad on a zoomed-in scale of its own.
    fig, axes = plt.subplots(
        1, len(targets), figsize=(5.5 * len(targets), 4.6),
        sharey=True, squeeze=False
    )

    for col, target in enumerate(targets):

        ax = axes[0, col]

        # One x grid per target, from every realisation scored at this horizon.
        # Deriving it per model would put the curves on slightly different
        # grids wherever the models were scored on different origin counts.
        at_horizon = (
            (scores.target == target) & (scores.horizon == HORIZON)
        ).to_numpy()

        grid = np.linspace(
            np.percentile(scores.actual.to_numpy()[at_horizon], 0.5),
            np.percentile(scores.actual.to_numpy()[at_horizon], 99.5),
            160
        )

        for model in by_target[target]:

            mask = at_horizon & (scores.model == model).to_numpy()

            delta = marginal_calibration(
                predictions[mask], levels, scores.actual.to_numpy()[mask], grid
            )

            ax.plot(
                grid, delta,
                color=colors[model],
                linewidth=1.8,
                label=model_label(model)
            )

        ax.axhline(0.0, color=BASELINE, linewidth=1.2, zorder=1)

        ax.set_title(
            target_label(target),
            loc="left", color=INK_PRIMARY, fontweight="bold", pad=8
        )

        ax.set_xlabel("x  (value of the variable)")

        if col == 0:
            ax.set_ylabel(r"$\bar{F}(x) - \hat{G}(x)$")

        ax.legend(
            frameon=False, fontsize=9,
            labelcolor=INK_SECONDARY, loc="lower right"
        )

        style_axes(ax)

    fig.suptitle(
        f"Marginal calibration, h = {HORIZON}  -  closer to zero is better",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK_PRIMARY
    )

    fig.text(
        0.01, 0.005,
        "Average predictive CDF minus the empirical CDF of the realisations. "
        "Positive means the forecasts put too much probability below x. "
        "Marginal calibration is necessary, not sufficient - read it beside "
        "the PIT histograms.",
        fontsize=8, color=INK_MUTED
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.92])

    path = os.path.join(FIGURE_FOLDER, "marginal_calibration.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():

    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    scores, predictions, levels = load()

    if len(scores) != len(predictions):
        raise ValueError(
            f"{len(scores)} score rows but {len(predictions)} prediction rows "
            "- re-run run_benchmarks.py so the two stay in step"
        )

    plot_pit(scores, levels)
    plot_marginal_calibration(scores, predictions, levels)


if __name__ == "__main__":
    main()
