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

from models.scoring import crps_from_quantiles  # noqa: E402

RESULTS_FOLDER = os.path.join(ROOT, "results")
FIGURE_FOLDER = os.path.join(ROOT, "figures")

HORIZON = 1
ALPHA = 0.5

MODEL_A = "qar"
MODEL_B = "historical_20y"

N_THETA = 320
N_BOOTSTRAP = 500
BLOCK_MONTHS = 12
BAND = 90               # central bootstrap band, %
TRIM = 0.5              # percentile of realisations trimmed off each end

SEED = 20260806

MODEL_LABELS = {
    "historical_20y": "Historical quantiles",
    "qar": "Quantile AR"
}

TARGET_LABELS = {
    "cpi_yoy": "CPI inflation, year-on-year (%)",
    "ipi_growth": "Industrial production, monthly growth (%)"
}

SCORE_TITLES = {
    "quantile": "Quantile $\\alpha=1/2$  (median forecast)",
    "expectile": "Expectile $\\alpha=1/2$  (mean forecast)",
    "brier": "Brier / CRPS  (full distribution)"
}

MODEL_COLORS = {"historical_20y": "#2a78d6", "qar": "#eb6834"}

DIFF_COLOR = "#4a3aa7"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

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
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9
})


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

def load_pairs(target):
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
    a = base[base.model == MODEL_A].set_index(keys).sort_index()
    b = base[base.model == MODEL_B].set_index(keys).sort_index()

    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    if not np.allclose(a.actual.to_numpy(), b.actual.to_numpy()):
        raise ValueError("paired rows disagree on the realisation")

    return {
        "actual": a.actual.to_numpy(),
        "origin": common.get_level_values("origin"),
        MODEL_A: {
            "median": a["median"].to_numpy(),
            "mean": a["mean"].to_numpy(),
            "pred": predictions[a.row.to_numpy()]
        },
        MODEL_B: {
            "median": b["median"].to_numpy(),
            "mean": b["mean"].to_numpy(),
            "pred": predictions[b.row.to_numpy()]
        },
        "levels": levels
    }


def score_matrices(pairs, theta):
    """Per-forecast elementary scores for both models, for all three types."""

    out = {}

    for model in [MODEL_A, MODEL_B]:

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

def style_axes(ax):

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)


def build_figure(target):

    rng = np.random.default_rng(SEED)

    pairs = load_pairs(target)
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

        for model in [MODEL_B, MODEL_A]:

            curve = matrices[model][score].mean(axis=0)

            top.plot(
                theta, curve,
                color=MODEL_COLORS[model],
                linewidth=1.8,
                label=MODEL_LABELS[model]
            )

            checks[(score, model)] = np.trapz(curve, theta)

        diff = matrices[MODEL_A][score] - matrices[MODEL_B][score]
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
            f"difference (qar - historical) · {verdict}",
            loc="left", color=INK_SECONDARY, fontsize=9, pad=6
        )

        for ax in (top, bottom):
            ax.set_xlim(
                np.percentile(actual, TRIM),
                np.percentile(actual, 100 - TRIM)
            )
            ax.set_xlabel(r"threshold $\theta$")
            style_axes(ax)

        if col == 0:
            top.set_ylabel(r"mean elementary score $\bar{S}_\theta$")
            bottom.set_ylabel("score difference")
            top.legend(
                frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY
            )

    fig.suptitle(
        f"Murphy diagram, h = {HORIZON}  -  {TARGET_LABELS[target]}",
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

    path = os.path.join(FIGURE_FOLDER, f"murphy_{target}.png")
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

    aggregated = pd.read_csv(
        os.path.join(RESULTS_FOLDER, "aggregated_by_horizon.csv")
    )

    for target in ["cpi_yoy", "ipi_growth"]:

        checks = build_figure(target)

        # Mixture identities: integrating the elementary scores over theta must
        # reproduce the scores computed independently in run_benchmarks.py.
        print(f"\n  {target}: integral of S_theta vs the reported score")

        for model in [MODEL_A, MODEL_B]:

            row = aggregated[
                (aggregated.target == target)
                & (aggregated.model == model)
                & (aggregated.horizon == HORIZON)
            ].iloc[0]

            # The Brier integral is checked against a *refined* CRPS, not the
            # one in the table. The tabulated CRPS is the 23-point pinball
            # approximation, which is documented to understate CRPS; refining
            # the quantile grid to 999 levels closes almost the whole gap, so
            # the discrepancy is grid coarseness rather than an error here.
            expected = {
                "quantile": (0.5 * row.mae_median_fc, "0.5 x MAE"),
                "expectile": (0.25 * row.rmse_mean_fc ** 2, "0.25 x MSE"),
                "brier": (refined_crps(target, model), "refined CRPS")
            }

            for score, (target_value, source) in expected.items():
                got = checks[(score, model)]
                tolerance = 0.02 * max(abs(target_value), 1e-9)
                flag = "ok " if abs(got - target_value) < tolerance else "OFF"
                print(f"    {flag} {model:15s} {score:10s} "
                      f"{got:8.4f} vs {target_value:8.4f}  ({source})")

            print(f"        tabulated CRPS {row.crps:.4f} understates the "
                  f"refined value by "
                  f"{100 * (1 - row.crps / refined_crps(target, model)):.1f}% "
                  "- the 23-point grid, identical for both models")


if __name__ == "__main__":
    main()
