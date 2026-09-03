"""Sundial: predictive quantiles of y_{t+h} from a univariate time series.

The model is loaded once and used to generate probabilistic forecasts from
the most recent LOOKBACK observations available at each forecast origin.

Interface:
    sundial(y, h=12) -> pd.Series

The returned Series contains one predictive quantile for each level in
QUANTILES, matching the interface used by the other forecasting models.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from config import LOOKBACK, N_SAMPLES, QUANTILES


MODEL_ID = "thuml/sundial-base-128m"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the Sundial model."""

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.eval()

    return model


def sundial(y, h=12, lookback=LOOKBACK, n_samples=N_SAMPLES,
            quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float).dropna()

    if len(y) < lookback:
        raise ValueError(
            f"Sundial requires at least {lookback} observations, "
            f"but only {len(y)} are available."
        )

    x = torch.tensor(
        y.iloc[-lookback:].values,
        dtype=torch.float32,
    ).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        samples = model.generate(
            x,
            max_new_tokens=h,
            num_samples=n_samples,
        )

    samples = samples.squeeze(0).cpu().numpy()

    forecast_samples = samples[:, h - 1]

    preds = [
        np.quantile(forecast_samples, q)
        for q in quantiles
    ]

    preds = np.sort(preds)

    return pd.Series(
        preds,
        index=quantiles,
        name="forecast",
    )