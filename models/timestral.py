"""Timestral - Mistral's pretrained time-series foundation model, zero-shot.

Timestral-209M used exactly as it ships: no fine-tuning and no per-series
adaptation. At each origin it sees the series' own past (up to LOOKBACK
points) and emits the predictive distribution for h = 1..H in one forward
pass. Timestral also supports multivariate inputs and covariates; here it is
fed one series at a time.

Its native grid is 21 quantiles [0.01, 0.05, 0.10, ..., 0.95, 0.99], exactly
the QUANTILES grid in config.toml, so nothing is interpolated.

The forecaster clamps forecasts at 0 by default whenever the context is
non-negative; that is switched off, since growth rates can turn negative
even after a window without negative values.

Interface matches the other models: timestral(y, h=12) -> pd.Series of
QUANTILES-indexed predictive quantiles for y_{t+h}.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from config import LOOKBACK, QUANTILES

MODEL_ID = "mistralai/Timestral-209M"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the Timestral model."""

    import torch
    from timestral import TimestralForecaster

    return TimestralForecaster.from_pretrained(
        MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu"
    )


def timestral(y, h=12, lookback=LOOKBACK, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float).dropna()
    context = y.iloc[-lookback:].to_numpy()

    model = load_model()
    forecast = model.forecast(
        context, prediction_length=int(h), clip_nonnegative=False
    )

    # (1, 1, h, 21) -> the quantile row for horizon h
    row = forecast.quantiles[0, 0, int(h) - 1].to("cpu").numpy().astype(float)

    preds = np.interp(quantiles, model.quantile_levels, row)

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")
