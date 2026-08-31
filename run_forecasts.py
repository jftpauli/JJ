"""Rolling-origin quantile forecasts for the CPI panel -> results/<model>_forecasts.csv."""

import pandas as pd

from config import (COUNTRIES, DATA_PATHS, FORECAST_HORIZONS,
                    FORECAST_START_DATE, QUANTILES, RESULTS_PATH)
from models import chronos2, historical_quantiles, quantile_ar

PANEL = DATA_PATHS["cpi"]

MODELS = {
    "historical": historical_quantiles,
    "qar": quantile_ar,
    "chronos2": chronos2,
}


def backtest(forecast, y, h):
    """Forecast y_{t+h} from every origin, on what was known at that origin.

    The forecaster sees y up to and including the origin and nothing after, so
    a model that estimates parameters re-estimates at every origin. `actual` is
    the realised value at the target date, NaN for origins whose target has not
    happened yet.
    """

    y = y.dropna()
    origins = y.index[y.index >= pd.Timestamp(FORECAST_START_DATE)]

    out = pd.DataFrame({t: forecast(y.loc[:t], h=h) for t in origins}).T
    out.columns = [f"q{tau}" for tau in QUANTILES]

    return out.assign(horizon=h, actual=y.shift(-h).reindex(out.index))


def main():

    panel = pd.read_csv(PANEL, index_col=0, parse_dates=True)[COUNTRIES]
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    for name, forecast in MODELS.items():

        path = RESULTS_PATH / f"{name}_forecasts.csv"

        pd.concat(
            {c: pd.concat([backtest(forecast, panel[c], h)
                           for h in FORECAST_HORIZONS])
             for c in panel},
            names=["country", "origin"]
        ).to_csv(path, float_format="%.6f")

        print("saved", path)


if __name__ == "__main__":
    main()
