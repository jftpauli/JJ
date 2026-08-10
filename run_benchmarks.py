"""Rolling-origin backtest of the benchmark models.

Expanding window, re-estimated every `refit_every` periods and applied at every
period in between - the usual compromise between a fully recursive scheme and
what a quantile regression grid costs to fit. Chronos-2 has nothing to estimate,
so the schedule does not bind it: it gets a fresh forward pass at every origin.

Targets are stationary transforms, not index levels. Quantile regression on an
integrated series is badly conditioned, and a density benchmark on a random
walk mostly measures the drift, so each target is a growth rate or a
year-on-year rate; see TARGETS below for the current set.

Not every model runs on every target. The model set is part of the target's
spec, so a model that only makes sense for one series - or that is only wanted
for one - is declared there rather than switched on inside the loop. Anything
downstream reads the model list off the scores file rather than assuming it.

Scores are written per (target, country, model, origin, horizon) so any
aggregation - by horizon, by country, excluding COVID - can be done downstream
without re-running the backtest.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    DEFAULT_QUANTILES,
    Chronos2,
    HistoricalQuantiles,
    QuantileAR,
    coverage,
    crps_from_quantiles,
    crps_skill_score,
    pit_from_quantiles,
    predictive_mean,
)


# ======================================================
# USER SETTINGS
# ======================================================

DATA_FOLDER = "data"
OUTPUT_FOLDER = "results"

H_MAX = 12
FIRST_ORIGIN = "2005-01-01"

# The two targets run at different frequencies, so anything measured in periods
# has to be set per target. h = 12 therefore means twelve *months* for CPI and
# twelve *quarters* (three years) for GDP - the horizon axes of the two are not
# the same axis, and tables should not be read across them.
#
# `models` names which benchmarks run on the target. Only CPI inflation carries
# the foundation model: the output target is mid-way through a change of source
# series, so putting Chronos-2 on it now would produce numbers that have to be
# thrown away. Add "chronos2" to that list once the series is settled.
TARGETS = {
    "cpi_yoy": {
        "file": "OECD_cpi_yoy_monthly_panel.csv",
        "freq": "M",
        "refit_every": 12,          # re-estimate annually
        "climatology_window": 240,  # 20 years
        "models": ["historical_20y", "qar", "chronos2"]
    },
    "gdp_growth": {
        "file": "OECD_gdp_growth_quarterly_panel.csv",
        "freq": "Q",
        "refit_every": 4,           # re-estimate annually
        "climatology_window": 80,   # 20 years
        "models": ["historical_20y", "qar"]
    }
}

# The 0.05-0.95 grid with the tails refined. The extra levels cost little (the
# quantile regression is linear in the number of levels) and they matter for
# calibration diagnostics: a PIT computed on a grid that stops at 0.05 can only
# ever return values in [0.05, 0.95], and a marginal calibration curve is
# meaningless in a tail the grid never reaches.
QUANTILES = np.unique(np.concatenate([
    [0.01, 0.025],
    DEFAULT_QUANTILES,
    [0.975, 0.99]
]))

# The reference every skill score is computed against.
REFERENCE_MODEL = "historical_20y"


def build_models(spec):
    """A fresh, unfitted set of benchmarks for one target.

    Only the models named in the target's `models` list are built, so a target
    that does not run the foundation model never pays to load its weights.
    """

    catalogue = {
        # Unconditional empirical quantiles over a rolling 20-year window - the
        # climatological reference. The window is given in the target's own
        # periods, so it is 20 years for both frequencies. No seasonal split:
        # the targets are already year-on-year or growth rates. The
        # expanding-window variant scored within 0.008 skill of this one
        # everywhere, so only one is kept.
        "historical_20y": lambda: HistoricalQuantiles(
            window=spec["climatology_window"], quantiles=QUANTILES
        ),
        "qar": lambda: QuantileAR(p=4, quantiles=QUANTILES),
        # Zero-shot: no fitting, and the full history at every origin. Note
        # that its pre-training window overlaps the evaluation sample, so it is
        # not out-of-sample in the way the other two are - see models/chronos2.py.
        "chronos2": lambda: Chronos2(quantiles=QUANTILES)
    }

    unknown = set(spec["models"]) - set(catalogue)

    if unknown:
        raise ValueError(f"unknown model(s) in target spec: {sorted(unknown)}")

    return {name: catalogue[name]() for name in spec["models"]}


# ======================================================
# TARGETS
# ======================================================

def load_targets():
    """Load the model-ready panels.

    Both files are already the stationary transform the models consume - the
    transformation lives in oecd_api.py, so nothing here has to know about it.
    """

    return {
        name: pd.read_csv(
            os.path.join(DATA_FOLDER, spec["file"]),
            index_col="TIME_PERIOD",
            parse_dates=True
        ).sort_index()
        for name, spec in TARGETS.items()
    }


# ======================================================
# BACKTEST
# ======================================================

def backtest_series(y, dates, first_origin, spec):
    """Score every model over the rolling origins of one series.

    `spec` is the target's entry in TARGETS: it fixes the model set, the
    re-estimation interval and the climatology window, all of which are counted
    in the target's own periods (months or quarters).

    Returns a list of per-(model, origin, horizon) score records.
    """

    origins = np.flatnonzero(
        (dates >= first_origin) & (np.arange(len(dates)) < len(dates) - 1)
    )

    records = []
    quantile_rows = []
    fitted = None
    skipped = 0

    for count, t in enumerate(origins):

        history = y[: t + 1]

        # Re-estimate on the expanding window every `refit_every` origins.
        if fitted is None or count % spec["refit_every"] == 0:
            fitted = {}
            for name, model in build_models(spec).items():
                try:
                    fitted[name] = model.fit(history)
                except ValueError:
                    fitted[name] = None

        horizons = np.arange(1, min(H_MAX, len(y) - 1 - t) + 1)

        if len(horizons) == 0:
            continue

        actuals = y[t + horizons]

        for name, model in fitted.items():

            if model is None:
                continue

            try:
                pred = model.predict_quantiles(len(horizons), history=history)
            except ValueError:
                # NaN at the forecast origin - the models refuse to impute.
                skipped += 1
                continue

            crps = crps_from_quantiles(actuals, pred, QUANTILES)
            hit_90 = coverage(actuals, pred, QUANTILES, 0.05, 0.95)
            hit_50 = coverage(actuals, pred, QUANTILES, 0.25, 0.75)
            pit = pit_from_quantiles(actuals, pred, QUANTILES)

            means = predictive_mean(pred, QUANTILES)

            i_med = int(np.argmin(np.abs(QUANTILES - 0.5)))
            i_lo = int(np.argmin(np.abs(QUANTILES - 0.05)))
            i_hi = int(np.argmin(np.abs(QUANTILES - 0.95)))

            for i, h in enumerate(horizons):

                if not np.isfinite(actuals[i]):
                    continue

                records.append({
                    "model": name,
                    "origin": dates[t],
                    "horizon": int(h),
                    "actual": actuals[i],
                    "median": pred[i, i_med],
                    "mean": means[i],
                    "crps": crps[i],
                    "covered_50": hit_50[i],
                    "covered_90": hit_90[i],
                    "width_90": pred[i, i_hi] - pred[i, i_lo],
                    "pit": pit[i]
                })

                # Full predictive quantiles, kept in step with `records` -
                # calibration diagnostics need the whole distribution, not the
                # scalar summaries.
                quantile_rows.append(pred[i])

    return records, quantile_rows, skipped


def series_scales(targets, first_origin):
    """A per-series scale factor, for pooling scores across different units.

    Standard deviation over the pre-evaluation sample only, so the
    normalisation carries no information from the evaluation period.

    The MASE denominator (mean absolute one-month change) is the more familiar
    choice but is wrong here: it measures short-run choppiness, not dispersion.
    Year-on-year inflation is smooth and persistent, so its mean absolute
    monthly change is tiny and scaled CRPS explodes to ~5, while noisy IPI
    growth lands at ~0.8 - a "pooled" score that is really just CPI. CRPS is a
    dispersion measure in the units of the variable, so the standard deviation
    is its natural scale, and a climatological forecast then sits near the same
    value (~0.56 for a Gaussian) whatever the series.
    """

    scales = {}

    for target_name, panel in targets.items():
        pre = panel.loc[panel.index < first_origin]
        for country in panel.columns:
            scales[(target_name, country)] = float(pre[country].std())

    return scales


def aggregate_by_horizon(scores):
    """Every evaluation metric, averaged over all countries and all origins.

    One row per (target, model, horizon): 12 rows per model per target, each
    averaging over the 4 countries and every forecast origin.
    """

    grouped = scores.groupby(["target", "model", "horizon"])

    table = grouped.apply(
        lambda g: pd.Series({
            "n_forecasts": len(g),
            "crps": g.crps.mean(),
            "crps_median": g.crps.median(),
            "crps_scaled": g.crps_scaled.mean(),
            "pinball": g.crps.mean() / 2.0,
            # Each functional is scored by the loss that elicits it: absolute
            # error for the median forecast, squared error for the mean
            # forecast (Gneiting 2011). Crossing them over compares a point
            # forecast against a target it was never aiming at.
            "mae_median_fc": (g.actual - g["median"]).abs().mean(),
            "medae_median_fc": (g.actual - g["median"]).abs().median(),
            "rmse_mean_fc": np.sqrt(((g.actual - g["mean"]) ** 2).mean()),
            "coverage_50": g.covered_50.mean(),
            "coverage_90": g.covered_90.mean(),
            "width_90": g.width_90.mean(),
            "pit_mean": g.pit.mean(),
            "pit_ks": stats.kstest(g.pit, "uniform").statistic
        }),
        include_groups=False
    )

    # Skill score against the reference model, horizon by horizon.
    reference = table.xs(REFERENCE_MODEL, level="model")["crps"]
    table["crps_skill"] = 1.0 - table["crps"] / table.index.droplevel(
        "model"
    ).map(reference)

    return table


def comparison_pairs(models):
    """Which (challenger, incumbent) pairs to run the paired comparison on.

    Every model is compared against the climatological reference, which is what
    a skill score means. On top of that, the foundation model is compared
    against the best estimated benchmark rather than only against climatology -
    beating a distribution that conditions on nothing is a low bar, and "does
    pre-training beat a quantile regression fitted on this series" is the
    question the exercise actually asks.
    """

    pairs = [
        (name, REFERENCE_MODEL) for name in models if name != REFERENCE_MODEL
    ]

    if "chronos2" in models and "qar" in models:
        pairs.append(("chronos2", "qar"))

    return pairs


def compare_models(scores, model_a="qar", model_b=REFERENCE_MODEL):
    """Paired comparison of two models, by horizon.

    Forecasts are paired on (target, country, origin, horizon) so the two
    models are always compared on exactly the same forecasting problem. That
    matters more than it sounds: comparing separately-averaged scores lets a
    model look better merely by having been evaluated on an easier subset.

    Three scoring functions, each consistent for the functional it scores:

        crps  - the whole predictive distribution
        ae    - absolute error of the median forecast
        se    - squared error of the mean forecast

    For each, the mean and the median of the paired difference (a minus b).
    Negative favours `model_a`. The median difference and the win rate are the
    robust readings: a handful of COVID months dominate every mean here, so a
    mean difference can have the opposite sign to what happens in a typical
    month.

    The sign test asks whether `model_a` wins more often than half the time.
    Its p-value assumes independent pairs, which these are not - four
    countries move together and, past h = 1, forecast errors overlap - so read
    it as descriptive, not as inference. A Diebold-Mariano test with HAC
    standard errors is the version that belongs in a paper.
    """

    keys = ["target", "country", "origin", "horizon"]

    a = scores[scores.model == model_a].set_index(keys)
    b = scores[scores.model == model_b].set_index(keys)

    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    paired = pd.DataFrame({
        "target": common.get_level_values("target"),
        "horizon": common.get_level_values("horizon"),
        "d_crps": (a.crps - b.crps).to_numpy(),
        "d_ae": (
            (a.actual - a["median"]).abs() - (b.actual - b["median"]).abs()
        ).to_numpy(),
        "d_se": (
            (a.actual - a["mean"]) ** 2 - (b.actual - b["mean"]) ** 2
        ).to_numpy()
    })

    def summarise(g):

        row = {"n_pairs": len(g)}

        for score in ["crps", "ae", "se"]:

            diff = g[f"d_{score}"].to_numpy()
            wins = int((diff < 0).sum())
            decided = int((diff != 0).sum())

            row[f"{score}_mean_diff"] = diff.mean()
            row[f"{score}_median_diff"] = np.median(diff)
            row[f"{score}_win_rate"] = wins / decided if decided else np.nan
            row[f"{score}_sign_p"] = (
                stats.binomtest(wins, decided, 0.5).pvalue
                if decided else np.nan
            )

        return pd.Series(row)

    return paired.groupby(["target", "horizon"]).apply(
        summarise, include_groups=False
    )


def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    targets = load_targets()
    first_origin = pd.Timestamp(FIRST_ORIGIN)

    all_records = []
    all_quantiles = []
    total_skipped = 0

    for target_name, panel in targets.items():

        for country in panel.columns:

            series = panel[country]
            y = series.to_numpy(dtype=float)
            dates = series.index.to_numpy()

            print(f"  {target_name:12s} {country} ...", flush=True)

            records, quantile_rows, skipped = backtest_series(
                y, dates, first_origin, TARGETS[target_name]
            )
            total_skipped += skipped

            for record in records:
                record["target"] = target_name
                record["country"] = country

            all_records.extend(records)
            all_quantiles.extend(quantile_rows)

    scores = pd.DataFrame(all_records)

    # Scale-free CRPS, so scores can be pooled across targets with different
    # units. Raw CRPS cannot: CPI inflation and IPI growth are not comparable.
    scales = series_scales(targets, first_origin)
    scores["crps_scaled"] = scores.crps / [
        scales[(t, c)] for t, c in zip(scores.target, scores.country)
    ]

    path = os.path.join(OUTPUT_FOLDER, "benchmark_scores.csv")
    scores.to_csv(path, index=False)

    # Row i of `predictions` is the predictive distribution behind row i of
    # benchmark_scores.csv.
    np.savez_compressed(
        os.path.join(OUTPUT_FOLDER, "predictive_quantiles.npz"),
        predictions=np.asarray(all_quantiles, dtype=float),
        quantile_levels=QUANTILES
    )

    by_horizon = aggregate_by_horizon(scores)
    horizon_path = os.path.join(OUTPUT_FOLDER, "aggregated_by_horizon.csv")
    by_horizon.round(6).to_csv(horizon_path)

    # ---------- summary ----------

    print("\n" + "=" * 64)
    print("Mean CRPS by target and model (lower is better)")
    print("=" * 64)
    print(
        scores.pivot_table(
            index="target", columns="model", values="crps", aggfunc="mean"
        ).round(4).to_string()
    )

    print(f"\nCRPS skill score vs '{REFERENCE_MODEL}' (>0 beats it, <0 worse)")
    mean_crps = scores.pivot_table(
        index="target", columns="model", values="crps", aggfunc="mean"
    )
    skill = mean_crps.apply(
        lambda column: crps_skill_score(column, mean_crps[REFERENCE_MODEL])
    )
    print(skill.round(3).to_string())

    # ---------- the by-horizon tables ----------

    metrics = [
        "n_forecasts", "crps", "crps_median", "crps_skill", "crps_scaled",
        "pinball", "mae_median_fc", "medae_median_fc", "rmse_mean_fc",
        "coverage_50", "coverage_90", "width_90", "pit_mean", "pit_ks"
    ]

    # Model sets differ by target, so the pairs are read off the results rather
    # than taken as the cross-product.
    for target_name in sorted(scores.target.unique()):

        present = sorted(scores[scores.target == target_name].model.unique())

        for model_name in present:

            print("\n" + "=" * 78)
            print(f"{target_name}  |  {model_name}   "
                  "(each row averages 4 countries x all origins)")
            print("=" * 78)
            print(
                by_horizon.loc[(target_name, model_name), metrics]
                .round(4).to_string()
            )

    # ---------- paired model comparison ----------

    pairs = comparison_pairs(sorted(scores.model.unique()))

    comparisons = pd.concat(
        {
            f"{model_a}_vs_{model_b}": compare_models(scores, model_a, model_b)
            for model_a, model_b in pairs
        },
        names=["comparison"]
    )

    comparison_path = os.path.join(OUTPUT_FOLDER, "model_comparison.csv")
    comparisons.round(6).to_csv(comparison_path)

    columns = [
        "n_pairs",
        "crps_mean_diff", "crps_median_diff", "crps_win_rate",
        "ae_mean_diff", "ae_median_diff", "ae_win_rate",
        "se_mean_diff", "se_median_diff", "se_win_rate"
    ]

    for model_a, model_b in pairs:

        block = comparisons.loc[f"{model_a}_vs_{model_b}"]

        for target_name in sorted(block.index.get_level_values("target").unique()):

            print("\n" + "=" * 78)
            print(f"{target_name}  |  paired: {model_a} minus {model_b} "
                  f"(negative favours {model_a})")
            print("=" * 78)
            print(block.loc[target_name][columns].round(4).to_string())

    # Pooling is only meaningful for models evaluated on every target. A model
    # run on one target only would otherwise be averaged over a different -
    # and here an easier - set of series than its rivals, and the pooled column
    # would compare it against nothing in particular.
    n_targets = scores.target.nunique()
    coverage_by_model = scores.groupby("model").target.nunique()
    pooled_models = sorted(coverage_by_model[coverage_by_model == n_targets].index)
    partial_models = sorted(set(coverage_by_model.index) - set(pooled_models))

    print("\n" + "=" * 78)
    print(f"Pooled across all {n_targets} targets and all "
          f"{scores.groupby(['target', 'country']).ngroups} series - scaled CRPS")
    print("(raw CRPS is not comparable across targets; this divides by each "
          "series'\n pre-2005 standard deviation, so they are commensurable)")
    print("=" * 78)
    print(
        scores[scores.model.isin(pooled_models)].pivot_table(
            index="horizon", columns="model", values="crps_scaled",
            aggfunc="mean"
        ).round(4).to_string()
    )

    if partial_models:
        print(f"\nNot pooled - evaluated on a subset of the targets: "
              f"{', '.join(partial_models)}. Read these per target instead; "
              "pooling them\nagainst models scored on a different set of "
              "series would not be a comparison.")

    print("\n90% interval coverage (nominal 0.90)")
    print(
        scores.pivot_table(
            index="target", columns="model", values="covered_90", aggfunc="mean"
        ).round(3).to_string()
    )

    print("\nExcluding COVID (origins and targets in 2020-2021)")
    calm = scores[
        (scores.origin.dt.year < 2020) | (scores.origin.dt.year > 2021)
    ]
    print(
        calm.pivot_table(
            index="target", columns="model", values="crps", aggfunc="mean"
        ).round(4).to_string()
    )

    if total_skipped:
        print(f"\nSkipped {total_skipped} origin-model pairs (NaN at origin).")

    print(f"\nScores written to {path}  ({len(scores):,} rows)")


if __name__ == "__main__":
    main()
