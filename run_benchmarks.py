"""Rolling-origin backtest of the benchmark models.

Expanding window, re-estimated every REFIT_EVERY months and applied at every
month in between - the usual compromise between a fully recursive scheme and
what a quantile regression grid costs to fit.

Targets are stationary transforms, not index levels. Quantile regression on an
integrated series is badly conditioned, and a density benchmark on a random
walk mostly measures the drift, so:

    cpi_yoy     year-on-year CPI inflation, %   (as published)
    ipi_growth  100 * dlog(industrial production), monthly %

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
REFIT_EVERY = 12

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


def build_models():
    """A fresh, unfitted set of benchmarks."""

    return {
        # Unconditional empirical quantiles over a rolling 20-year window - the
        # climatological reference. No seasonal split: the targets are already
        # year-on-year or growth rates. The expanding-window variant scored
        # within 0.008 skill of this one everywhere, so only one is kept.
        "historical_20y": HistoricalQuantiles(window=240, quantiles=QUANTILES),
        "qar": QuantileAR(p=4, quantiles=QUANTILES)
    }


# ======================================================
# TARGETS
# ======================================================

def load_targets():
    """Load the model-ready panels.

    Both files are already the stationary transform the models consume - the
    transformation lives in oecd_api.py, so nothing here has to know about it.
    """

    files = {
        "cpi_yoy": "OECD_cpi_yoy_monthly_panel.csv",
        "ipi_growth": "OECD_ipi_growth_monthly_panel.csv"
    }

    return {
        name: pd.read_csv(
            os.path.join(DATA_FOLDER, filename),
            index_col="TIME_PERIOD",
            parse_dates=True
        ).sort_index()
        for name, filename in files.items()
    }


# ======================================================
# BACKTEST
# ======================================================

def backtest_series(y, dates, first_origin):
    """Score every model over the rolling origins of one series.

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

        # Re-estimate on the expanding window every REFIT_EVERY origins.
        if fitted is None or count % REFIT_EVERY == 0:
            fitted = {}
            for name, model in build_models().items():
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
                y, dates, first_origin
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

    for target_name in sorted(scores.target.unique()):
        for model_name in sorted(scores.model.unique()):

            print("\n" + "=" * 78)
            print(f"{target_name}  |  {model_name}   "
                  "(each row averages 4 countries x all origins)")
            print("=" * 78)
            print(
                by_horizon.loc[(target_name, model_name), metrics]
                .round(4).to_string()
            )

    # ---------- paired model comparison ----------

    comparison = compare_models(scores)
    comparison_path = os.path.join(OUTPUT_FOLDER, "model_comparison.csv")
    comparison.round(6).to_csv(comparison_path)

    for target_name in sorted(scores.target.unique()):

        print("\n" + "=" * 78)
        print(f"{target_name}  |  paired: qar minus {REFERENCE_MODEL} "
              "(negative favours qar)")
        print("=" * 78)
        print(
            comparison.loc[target_name][[
                "n_pairs",
                "crps_mean_diff", "crps_median_diff", "crps_win_rate",
                "ae_mean_diff", "ae_median_diff", "ae_win_rate",
                "se_mean_diff", "se_median_diff", "se_win_rate"
            ]].round(4).to_string()
        )

    print("\n" + "=" * 78)
    print("Pooled across both targets and all 8 series - scaled CRPS")
    print("(raw CRPS is not comparable across targets; this divides by each "
          "series'\n pre-2005 standard deviation, so the two are commensurable)")
    print("=" * 78)
    print(
        scores.pivot_table(
            index="horizon", columns="model", values="crps_scaled",
            aggfunc="mean"
        ).round(4).to_string()
    )

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
