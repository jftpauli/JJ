"""Run rolling-origin quantile forecasts for ONE model, invoked by run_forecasts.py
in the model's own venv -> results/<model>_<indicator>_forecasts.csv.

Usage: python forecast_worker.py <model_name>
"""

import sys

import pandas as pd

from config import (COUNTRIES, DATA_PATHS, FORECAST_HORIZONS,
                    FORECAST_START_DATE, INDICATORS, QUANTILES, RESULTS_PATH)

AVAILABLE_MODELS = {
    "historical": lambda: __import__("models.historical_quantiles", fromlist=["historical_quantiles"]).historical_quantiles,
    "qar": lambda: __import__("models.qar", fromlist=["quantile_ar"]).quantile_ar,
    "chronos2": lambda: __import__("models.chronos2", fromlist=["chronos2"]).chronos2,
    "sundial": lambda: __import__("models.sundial", fromlist=["sundial"]).sundial,
    "timesfm": lambda: __import__("models.timesfm", fromlist=["timesfm"]).timesfm,
}


def print_progress(label, done, total):
    """Single-line progress bar, no external dependency."""

    width = 30
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if done == total else ""
    print(f"\r{label} [{bar}] {done}/{total}", end=end, flush=True)


def backtest(forecast, y, h):
    """Forecast y_{t+h} from every origin, on what was known at that origin.

    The forecaster sees y up to and including the origin and nothing after, so
    a model that estimates parameters re-estimates at every origin. `actual` is
    the realised value at the target date, NaN for origins whose target has not
    happened yet.
    """

    y = y.dropna()
    # drop the last h origins so every target date has an actual value on record
    origins = y.index[y.index >= pd.Timestamp(FORECAST_START_DATE)][:-h]

    if len(origins) == 0:
        columns = [f"q{tau}" for tau in QUANTILES]
        return pd.DataFrame(
            columns=columns + ["horizon", "actual"]
        ).rename_axis("origin")

    out = pd.DataFrame({t: forecast(y.loc[:t], h=h) for t in origins}).T
    out.columns = [f"q{tau}" for tau in QUANTILES]

    return out.assign(horizon=h, actual=y.shift(-h).reindex(out.index))


def country_forecasts(forecast, series):
    """Combine available horizons, allowing a country to have no origins."""

    frames = [
        backtest(forecast, series, h)
        for h in FORECAST_HORIZONS
    ]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames)


def main(model_name):

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown forecast model: {model_name}")

    forecast = AVAILABLE_MODELS[model_name]()

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    for indicator in INDICATORS:

        panel = pd.read_csv(DATA_PATHS[indicator], index_col=0, parse_dates=True)[COUNTRIES]

        path = RESULTS_PATH / f"{model_name}_{indicator}_forecasts.csv"

        country_results = {}
        for i, c in enumerate(panel, start=1):
            print_progress(f"{model_name}/{indicator}", i, len(panel.columns))
            country_results[c] = country_forecasts(forecast, panel[c])

        country_results = {
            c: result
            for c, result in country_results.items()
            if not result.empty
        }

        pd.concat(
            country_results,
            names=["country", "origin"]
        ).to_csv(path, float_format="%.6f")

        print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1])
