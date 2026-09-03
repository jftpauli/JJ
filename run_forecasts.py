"""Rolling-origin quantile forecasts for each INDICATORS panel -> results/<model>_<indicator>_forecasts.csv."""

import os

import pandas as pd

from config import (COUNTRIES, DATA_PATHS, FORECAST_HORIZONS,
                    FORECAST_START_DATE, INDICATORS, QUANTILES, RESULTS_PATH)

# Import lazily per model so chronos2/sundial's mutually incompatible deps are
# only loaded when that model is actually selected to run.
AVAILABLE_MODELS = {
    "historical": lambda: __import__("models.historical_quantiles", fromlist=["historical_quantiles"]).historical_quantiles,
    "qar": lambda: __import__("models.qar", fromlist=["quantile_ar"]).quantile_ar,
    "chronos2": lambda: __import__("models.chronos2", fromlist=["chronos2"]).chronos2,
    "sundial": lambda: __import__("models.sundial", fromlist=["sundial"]).sundial,
    "timesfm": lambda: __import__("models.timesfm", fromlist=["timesfm"]).timesfm,
}
# Select models to run here -> check venv requirements! .\environments\.venv-chronos\Scripts\Activate.ps1
MODEL_NAMES = tuple(
    name.strip()
    for name in os.environ.get("FORECAST_MODELS", "historical,qar,chronos2").split(",")
    if name.strip()
)

unknown_models = set(MODEL_NAMES) - AVAILABLE_MODELS.keys()
if unknown_models:
    raise ValueError(f"Unknown forecast model(s): {', '.join(sorted(unknown_models))}")

MODELS = {name: AVAILABLE_MODELS[name]() for name in MODEL_NAMES}


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

    out = pd.DataFrame({t: forecast(y.loc[:t], h=h) for t in origins}).T
    out.columns = [f"q{tau}" for tau in QUANTILES]

    return out.assign(horizon=h, actual=y.shift(-h).reindex(out.index))


def main():

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    for indicator in INDICATORS:

        panel = pd.read_csv(DATA_PATHS[indicator], index_col=0, parse_dates=True)[COUNTRIES]

        for name, forecast in MODELS.items():

            path = RESULTS_PATH / f"{name}_{indicator}_forecasts.csv"

            pd.concat(
                {c: pd.concat([backtest(forecast, panel[c], h)
                               for h in FORECAST_HORIZONS])
                 for c in panel},
                names=["country", "origin"]
            ).to_csv(path, float_format="%.6f")

            print("saved", path)


if __name__ == "__main__":
    main()
