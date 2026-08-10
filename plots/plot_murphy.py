"""Murphy diagrams: is the ranking robust to the choice of scoring function?

Ehm, Gneiting, Jordan & Krueger (2016, JRSS-B). Every consistent scoring
function for a functional is a mixture of *elementary* scores indexed by a
threshold theta. Plotting the mean elementary score against theta therefore
shows the ranking under *every* consistent loss at once:

  * one curve below the other everywhere -> that model dominates, and no choice
    of consistent scoring function can reverse it;
  * curves that cross -> the ranking genuinely depends on the loss, and any
    single headline number is a choice, not a finding.

That second case is not hypothetical here. On IPI growth at h = 1 the mean and
the median of the paired CRPS difference have opposite signs, so the question
"which model is better" has no loss-free answer until this diagram is drawn.

Three functionals, three elementary scores - they are not interchangeable, and
each is paired with the forecast it actually elicits:

  quantile 1/2   median forecast
      S_theta(x,y) = a 1{x <= theta < y} + (1-a) 1{y <= theta < x},  a = 1/2
      integrates over theta to the pinball loss

  expectile 1/2  mean forecast
      same indicators, weighted by |y - theta|
      integrates to the squared error (up to a factor)

  Brier          the whole predictive distribution
      S_theta(F,y) = (F(theta) - 1{y <= theta})^2
      integrates to the CRPS

Uncertainty is a moving-block bootstrap over forecast origins, resampling all
four countries together: origins are blocked because scores are serially
dependent, countries are kept together because they co-move. Bands are
pointwise, so they say nothing about dominance *uniformly* over theta - a curve
that stays inside its rival's band at every theta is suggestive, not a test.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths are anchored to the repository root rather than the working directory,
# so the script runs the same from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.scoring import crps_from_quantiles  # noqa: E402
# The same helper the backtest uses to decide which models are worth pairing,
# so the diagrams and the printed comparison tables never drift apart.
from run_benchmarks import comparison_pairs  # noqa: E402
from style import (  # noqa: E402
    BASELINE,
    DIFF_COLOR,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    apply_rcparams,
    model_colors,
    model_label,
    style_axes,
    target_label,
)

RESULTS_FOLDER = os.path.join(ROOT, "results")
FIGURE_FOLDER = os.path.join(ROOT, "figures")

HORIZON = 1
ALPHA = 0.5

N_THETA = 320
N_BOOTSTRAP = 500
BLOCK_MONTHS = 12
BAND = 90               # central bootstrap band, %
TRIM = 0.5              # percentile of realisations trimmed off each end

SEED = 20260806

SCORE_TITLES = {
    "quantile": "Quantile $\\alpha=1/2$  (median forecast)",
    "expectile": "Expectile $\\alpha=1/2$  (mean forecast)",
    "brier": "Brier / CRPS  (full distribution)"
}

apply_rcparams(tick_size=8, axes_title_size=10, axes_label_size=9)


# ======================================================
# ELEMENTARY SCORES
# ======================================================

def elementary_quantile(point, actual, theta, alpha=ALPHA):
    """S_theta for the alpha-quantile. Rows are forecasts, columns theta."""

    x = np.asarray(point)[:, None]
    y = np.asarray(actual)[:, None]
    t = np.asarray(theta)[None, :]

    below = (x <= t) & (t < y)
    above = (y <= t) & (t < x)

    return alpha * below + (1.0 - alpha) * above


def elementary_expectile(point, actual, theta, alpha=ALPHA):
    """S_theta for the alpha-expectile (alpha = 1/2 gives the mean)."""

    y = np.asarray(actual)[:, None]
    t = np.asarray(theta)[None, :]

    return np.abs(y - t) * elementary_quantile(point, actual, theta, alpha)


def elementary_brier(cdf_at_theta, actual, theta):
    """S_theta = (F(theta) - 1{y <= theta})^2; integrates to the CRPS."""

    y = np.asarray(actual)[:, None]
    t = np.asarray(theta)[None, :]

    return (cdf_at_theta - (y <= t).astype(float)) ** 2


def cdf_on_grid(predictions, levels, theta):
    """Predictive CDF evaluated on the theta grid, one row per forecast.

    Outside the stored quantile range the CDF is taken as 0 / 1 - i.e. the
    predictive distribution is treated as supported within its 1%-99%
    quantiles. That understates the tails slightly and so understates CRPS,
    identically for both models, so the comparison stays fair.
    """

    out = np.empty((len(predictions), len(theta)))

    for i in range(len(predictions)):
        out[i] = np.interp(
            theta, predictions[i], levels, left=0.0, right=1.0
        )

    return out


# ======================================================
# DATA
# ======================================================

def load_pairs(target, model_a, model_b):
    """Forecasts of both models for one target, aligned row by row."""

    scores = pd.read_csv(
        os.path.join(RESULTS_FOLDER, "benchmark_scores.csv"),
        parse_dates=["origin"]
    )

    bundle = np.load(os.path.join(RESULTS_FOLDER, "predictive_quantiles.npz"))
    predictions, levels = bundle["predictions"], bundle["quantile_levels"]

    if len(scores) != len(predictions):
        raise ValueError(
            "score rows and prediction rows are out of step - re-run "
            "run_benchmarks.py"
        )

    scores = scores.assign(row=np.arange(len(scores)))

    base = scores[
        (scores.target == target) & (scores.horizon == HORIZON)
    ]

    keys = ["country", "origin"]
    a = base[base.model == model_a].set_index(keys).sort_index()
    b = base[base.model == model_b].set_index(keys).sort_index()

    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    if not np.allclose(a.actual.to_numpy(), b.actual.to_numpy()):
        raise ValueError("paired rows disagree on the realisation")

    return {
        "actual": a.actual.to_numpy(),
        "origin": common.get_level_values("origin"),
        "models": (model_a, model_b),
        model_a: {
            "median": a["median"].to_numpy(),
            "mean": a["mean"].to_numpy(),
            "pred": predictions[a.row.to_numpy()]
        },
        model_b: {
            "median": b["median"].to_numpy(),
            "mean": b["mean"].to_numpy(),
            "pred": predictions[b.row.to_numpy()]
        },
        "levels": levels
    }


def score_matrices(pairs, theta):
    """Per-forecast elementary scores for both models, for all three types."""

    out = {}

    for model in pairs["models"]:

        block = pairs[model]

        out[model] = {
            "quantile": elementary_quantile(
                block["median"], pairs["actual"], theta
            ),
            "expectile": elementary_expectile(
                block["mean"], pairs["actual"], theta
            ),
            "brier": elementary_brier(
                cdf_on_grid(block["pred"], pairs["levels"], theta),
                pairs["actual"], theta
            )
        }

    return out


# ======================================================
# BOOTSTRAP
# ======================================================

def block_bootstrap_band(diff, origins, rng):
    """Pointwise band for the mean score difference.

    Moving blocks over distinct origins; every country at a resampled origin
    travels with it, so both the serial and the cross-sectional dependence are
    respected. Resampling individual forecasts would ignore both and give
    bands that are far too tight.
    """

    unique_origins = pd.DatetimeIndex(sorted(set(origins)))
    position = {o: i for i, o in enumerate(unique_origins)}

    rows_by_origin = [[] for _ in unique_origins]

    for row, origin in enumerate(origins):
        rows_by_origin[position[origin]].append(row)

    rows_by_origin = [np.array(r) for r in rows_by_origin]

    n_origins = len(unique_origins)
    n_blocks = int(np.ceil(n_origins / BLOCK_MONTHS))
    max_start = max(n_origins - BLOCK_MONTHS, 0)

    draws = np.empty((N_BOOTSTRAP, diff.shape[1]))

    for b in range(N_BOOTSTRAP):

        starts = rng.integers(0, max_start + 1, size=n_blocks)

        picked = np.concatenate([
            np.arange(s, min(s + BLOCK_MONTHS, n_origins)) for s in starts
        ])[:n_origins]

        rows = np.concatenate([rows_by_origin[i] for i in picked])
        draws[b] = diff[rows].mean(axis=0)

    lower = (100 - BAND) / 2

    return (
        np.percentile(draws, lower, axis=0),
        np.percentile(draws, 100 - lower, axis=0)
    )


# ======================================================
# PLOT
# ======================================================

def build_figure(target, model_a, model_b):

    rng = np.random.default_rng(SEED)

    colors = model_colors([model_b, model_a])

    pairs = load_pairs(target, model_a, model_b)
    actual = pairs["actual"]

    # The grid spans the full outcome range so the mixture identities hold; the
    # axes are trimmed afterwards only for readability.
    low, high = actual.min(), actual.max()
    pad = 0.02 * (high - low)
    theta = np.linspace(low - pad, high + pad, N_THETA)

    matrices = score_matrices(pairs, theta)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))

    checks = {}

    for col, score in enumerate(["quantile", "expectile", "brier"]):

        top, bottom = axes[0, col], axes[1, col]

        for model in [model_b, model_a]:

            curve = matrices[model][score].mean(axis=0)

            top.plot(
                theta, curve,
                color=colors[model],
                linewidth=1.8,
                label=model_label(model)
            )

            checks[(score, model)] = np.trapz(curve, theta)

        diff = matrices[model_a][score] - matrices[model_b][score]
        mean_diff = diff.mean(axis=0)

        lower, upper = block_bootstrap_band(diff, pairs["origin"], rng)

        bottom.fill_between(
            theta, lower, upper,
            color=DIFF_COLOR, alpha=0.16, linewidth=0
        )

        bottom.plot(theta, mean_diff, color=DIFF_COLOR, linewidth=1.8)
        bottom.axhline(0.0, color=BASELINE, linewidth=1.2)

        top.set_title(
            SCORE_TITLES[score], loc="left",
            color=INK_PRIMARY, fontweight="bold", pad=6
        )

        # Only count crossings that are materially non-zero. At the edges of
        # the grid both elementary scores vanish, so the raw difference
        # oscillates in floating-point noise and a naive sign count reports
        # crossings for cases that are visually clean dominance.
        scale = np.max(np.abs(mean_diff))
        material = mean_diff[np.abs(mean_diff) > 0.01 * scale]
        crossings = int(np.sum(np.diff(np.sign(material)) != 0))

        verdict = (
            "no material crossing - dominance" if crossings == 0
            else f"{crossings} crossing{'' if crossings == 1 else 's'} - "
                 "ranking depends on the loss"
        )

        bottom.set_title(
            f"difference ({model_a} - {model_b}) · {verdict}",
            loc="left", color=INK_SECONDARY, fontsize=9, pad=6
        )

        for ax in (top, bottom):
            ax.set_xlim(
                np.percentile(actual, TRIM),
                np.percentile(actual, 100 - TRIM)
            )
            ax.set_xlabel(r"threshold $\theta$")
            style_axes(ax, grid_width=0.7)

        if col == 0:
            top.set_ylabel(r"mean elementary score $\bar{S}_\theta$")
            bottom.set_ylabel("score difference")
            top.legend(
                frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY
            )

    fig.suptitle(
        f"Murphy diagram, h = {HORIZON}  -  {target_label(target)}  -  "
        f"{model_label(model_a)} vs {model_label(model_b)}",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK_PRIMARY
    )

    fig.text(
        0.01, 0.005,
        "Lower is better. A curve below its rival at every theta beats it under "
        "every consistent scoring function for that functional; crossings mean "
        f"the ranking depends on the loss. Shaded: {BAND}% pointwise "
        f"moving-block bootstrap ({BLOCK_MONTHS}-month blocks over origins, "
        "countries resampled together).",
        fontsize=8, color=INK_MUTED
    )

    fig.tight_layout(rect=[0, 0.035, 1, 0.95])

    path = os.path.join(
        FIGURE_FOLDER, f"murphy_{target}_{model_a}_vs_{model_b}.png"
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", path)

    return checks


def refined_crps(target, model, n_levels=999):
    """Mean CRPS on a quantile grid refined by interpolation.

    The stored 23-level grid is coarse enough that the averaged-pinball CRPS
    understates the true value by around a tenth. Interpolating each quantile
    function onto a dense grid removes most of that discretisation error and
    gives something the Brier integral can be checked against.
    """

    scores = pd.read_csv(
        os.path.join(RESULTS_FOLDER, "benchmark_scores.csv")
    )

    bundle = np.load(os.path.join(RESULTS_FOLDER, "predictive_quantiles.npz"))
    predictions, levels = bundle["predictions"], bundle["quantile_levels"]

    mask = (
        (scores.target == target)
        & (scores.horizon == HORIZON)
        & (scores.model == model)
    ).to_numpy()

    pred = predictions[mask]
    actual = scores.actual.to_numpy()[mask]

    dense = np.linspace(levels[0], levels[-1], n_levels)
    fine = np.array([np.interp(dense, levels, pred[i]) for i in range(len(pred))])

    return float(crps_from_quantiles(actual, fine, dense).mean())


def main():

    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    scores = pd.read_csv(os.path.join(RESULTS_FOLDER, "benchmark_scores.csv"))

    aggregated = pd.read_csv(
        os.path.join(RESULTS_FOLDER, "aggregated_by_horizon.csv")
    )

    # Targets and model pairs come from the results file, so adding a model to
    # one target in run_benchmarks.py produces its diagrams here without any
    # edit - and a target that does not run that model does not get an empty
    # figure.
    for target in sorted(scores.target.unique()):

        present = sorted(scores[scores.target == target].model.unique())

        for model_a, model_b in comparison_pairs(present):

            checks = build_figure(target, model_a, model_b)

            # Mixture identities: integrating the elementary scores over theta
            # must reproduce the scores computed independently in
            # run_benchmarks.py.
            print(f"\n  {target}, {model_a} vs {model_b}: "
                  "integral of S_theta vs the reported score")

            report_checks(target, checks, [model_a, model_b], aggregated)


def report_checks(target, checks, models, aggregated):
    """Print the mixture-identity reconciliation for one figure.

    A caveat on reading the numbers: `checks` is computed on the *paired*
    subset (origins where both models produced a forecast) while the tabulated
    scores average over each model's own set. Where the two sets differ - only
    at origins with a NaN in the quantile regression's lags - the reconciliation
    is comparing slightly different samples, which is worth a few tenths of a
    percent, not a failed check.
    """

    for model in models:

        row = aggregated[
            (aggregated.target == target)
            & (aggregated.model == model)
            & (aggregated.horizon == HORIZON)
        ].iloc[0]

        refined = refined_crps(target, model)

        # The Brier integral is checked against a *refined* CRPS, not the one
        # in the table. The tabulated CRPS is the 23-point pinball
        # approximation, which is documented to understate CRPS; refining the
        # quantile grid to 999 levels closes almost the whole gap, so the
        # discrepancy is grid coarseness rather than an error here.
        expected = {
            "quantile": (0.5 * row.mae_median_fc, "0.5 x MAE"),
            "expectile": (0.25 * row.rmse_mean_fc ** 2, "0.25 x MSE"),
            "brier": (refined, "refined CRPS")
        }

        for score, (target_value, source) in expected.items():
            got = checks[(score, model)]
            tolerance = 0.02 * max(abs(target_value), 1e-9)
            flag = "ok " if abs(got - target_value) < tolerance else "OFF"
            print(f"    {flag} {model:15s} {score:10s} "
                  f"{got:8.4f} vs {target_value:8.4f}  ({source})")

        print(f"        tabulated CRPS {row.crps:.4f} understates the "
              f"refined value by {100 * (1 - row.crps / refined):.1f}% "
              "- the 23-point grid, identical for every model")


if __name__ == "__main__":
    main()
