"""Moirai 2.0 - the pretrained time-series foundation model, zero-shot.

Salesforce's Moirai-2.0-R-small used exactly as it ships: no fine-tuning and
no per-series adaptation. At each origin it sees the series' own past (up to
LOOKBACK points) and the quantile head emits the predictive distribution for
h = 1..H in one forward pass.

As with TimesFM, the quantile head's native grid is the fixed deciles
[0.1, 0.2, ..., 0.9]. Levels in QUANTILES outside that grid (0.01, 0.05, 0.95,
0.99) are linearly interpolated between the nearest deciles and clamped (not
extrapolated) beyond 0.1/0.9 - so the extreme tails are flatter than a model
with a native fit at those levels.

Interface matches the other models: moirai(y, h=12) -> pd.Series of
QUANTILES-indexed predictive quantiles for y_{t+h}.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from config import LOOKBACK, QUANTILES

MODEL_ID = "Salesforce/moirai-2.0-R-small"


@lru_cache(maxsize=None)
def load_model(h, context_length=LOOKBACK):
    """Load and cache Moirai 2.0 for one prediction length."""

    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    return Moirai2Forecast(
        module=Moirai2Module.from_pretrained(MODEL_ID),
        prediction_length=h,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )


def moirai(y, h=12, lookback=LOOKBACK, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float).dropna()
    context = y.iloc[-lookback:].to_numpy()

    model = load_model(int(h))
    # (1, n_native_quantiles, h) -> the decile column for horizon h
    deciles = model.predict(past_target=[context[:, None]])[0, :, int(h) - 1]

    preds = np.interp(quantiles, model.module.quantile_levels, deciles)

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")
