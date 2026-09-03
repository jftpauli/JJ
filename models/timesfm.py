"""TimesFM 2.5 - the pretrained time-series foundation model, zero-shot.

Google's TimesFm-2.5 (200M parameters) used exactly as it ships: no
fine-tuning and no per-series adaptation. At each origin it sees the series'
own past (up to LOOKBACK points) and the quantile head emits the predictive
distribution for the requested horizon in one forward pass.

The quantile head's native grid is the fixed deciles [0.1, 0.2, ..., 0.9] plus
the mean - it does not accept arbitrary quantile_levels the way Chronos-2
does. Levels in QUANTILES outside that grid (0.01, 0.05, 0.95, 0.99) are
therefore linearly interpolated between the nearest deciles, and clamped
(not extrapolated) beyond 0.1/0.9 - so the extreme tails are flatter than a
model with a native fit at those levels.

Interface matches the other models: timesfm(y, h=12) -> pd.Series of
QUANTILES-indexed predictive quantiles for y_{t+h}.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS, LOOKBACK, QUANTILES

MODEL_ID = "google/timesfm-2.5-200m-pytorch"

# The native quantile head grid: index 0 is the mean, 1..9 are deciles 0.1..0.9.
NATIVE_QUANTILES = np.arange(1, 10) / 10.0


@lru_cache(maxsize=1)
def load_model(max_context=LOOKBACK, max_horizon=max(FORECAST_HORIZONS)):
    """Load, compile and cache the TimesFM 2.5 model."""

    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    model.compile(timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    ))

    return model


def timesfm(y, h=12, lookback=LOOKBACK, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float).dropna()
    context = y.iloc[-lookback:].to_numpy()

    model = load_model()
    _, quantile_forecast = model.forecast(horizon=int(h), inputs=[context])

    # (1, h, 10) -> the decile row for horizon h, dropping the leading mean
    deciles = quantile_forecast[0, int(h) - 1, 1:]

    preds = np.interp(quantiles, NATIVE_QUANTILES, deciles)

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")
