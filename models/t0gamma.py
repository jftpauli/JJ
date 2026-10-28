"""t0-gamma - The Forecasting Company's pretrained time-series foundation model, zero-shot.

t0-gamma (~256M parameters) used exactly as it ships: no fine-tuning and no
per-series adaptation. At each origin it sees the series' own past (up to
LOOKBACK points) and emits the predictive distribution for h = 1..H in one
forward pass (horizons up to 1024 are decoded without autoregressive rollout).
t0-gamma also supports multivariate inputs and known-future covariates; here
it is fed one series at a time.

Its native grid is 21 quantiles [0.01, 0.05, 0.10, ..., 0.95, 0.99], exactly
the QUANTILES grid in config.toml, so nothing is interpolated.

The optional post-processing (symmetric averaging, clipping at 0, snapping to
integers) is left off, its default: the raw model output is scored, and
growth rates can turn negative even after a window without negative values.

Interface matches the other models: t0gamma(y, h=12) -> pd.Series of
QUANTILES-indexed predictive quantiles for y_{t+h}.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from config import LOOKBACK, QUANTILES

MODEL_ID = "theforecastingcompany/t0-gamma"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the t0-gamma model."""

    import torch
    from t0 import T0Forecaster

    model = T0Forecaster.from_pretrained(MODEL_ID)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    return model


def t0gamma(y, h=12, lookback=LOOKBACK, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float).dropna()
    context = y.iloc[-lookback:].to_numpy(dtype=np.float32)[None, :]

    forecast = load_model().predict(
        context,
        horizon=int(h),
        quantile_levels=[float(tau) for tau in quantiles],
    )

    # (1, h, n_quantiles) -> the quantile row for horizon h
    row = forecast.quantiles[0, int(h) - 1].to("cpu").numpy().astype(float)

    return pd.Series(np.sort(row), index=quantiles, name="forecast")
