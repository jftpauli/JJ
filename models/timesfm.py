"""TimesFM 3.0 - the pretrained time-series foundation model, zero-shot.

Google's TimesFM-3.0 (~330M parameters) used exactly as it ships: no
fine-tuning and no per-series adaptation. At each origin it sees the series'
own past (up to LOOKBACK points) and the quantile head emits the predictive
distribution for h = 1..H in one forward pass. TimesFM 3 also supports
multivariate inputs (variate attention); here it is fed one series at a time.

The quantile head's native grid is the fixed deciles [0.1, 0.2, ..., 0.9] - it
does not accept arbitrary quantile_levels the way Chronos-2 does. Levels in
QUANTILES outside that grid (0.01, 0.05, 0.95, 0.99) are therefore linearly
interpolated between the nearest deciles, and clamped (not extrapolated)
beyond 0.1/0.9 - so the extreme tails are flatter than a model with a native
fit at those levels.

Interface matches the other models: timesfm(y, h=12) -> pd.Series of
QUANTILES-indexed predictive quantiles for y_{t+h}.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from config import LOOKBACK, QUANTILES

MODEL_ID = "google/timesfm-3.0-pytorch"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the TimesFM 3.0 model."""

    import timesfm

    return timesfm.TimesFM3Forecaster.from_pretrained(MODEL_ID)


def timesfm(y, h=12, lookback=LOOKBACK, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float).dropna()
    context = y.iloc[-lookback:].to_numpy()

    model = load_model()
    forecast = model.predict(
        context, horizon=int(h), return_quantiles=True,
        use_symmetric_averaging=True,
    )

    # (h, 9) -> the decile row for horizon h
    deciles = forecast.quantiles[int(h) - 1]

    preds = np.interp(quantiles, model.config.quantiles, deciles)

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")
