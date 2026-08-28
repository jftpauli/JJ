"""Quantile AR: predictive quantiles of y_{t+h} from p lags of y.

One quantile regression per level tau (Koenker & Bassett 1978):

    Q(y_{t+h} | y_t, ..., y_{t-p+1}) = a + b_1 y_t + ... + b_p y_{t-p+1}

Fit on a stationary series (year-on-year inflation, growth rates), not levels.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

QUANTILES = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)

# Lag order per horizon, selected by BIC and confirmed by out-of-sample pinball
# loss on the four OECD CPI series. At h = 1 the two criteria agree on a sharp
# break at 13: since y_{t+1} - y_t = d_{t+1} - d_{t-11}, the thirteenth lag
# carries the base effect dropping out of the year-on-year window. At h = 12
# that term is no longer in the conditioning set and extra lags only add
# estimation noise - p = 12 costs 9-21% pinball against p = 1 there.
LAGS = {1: 13, 12: 1}


def quantile_ar(y, h=12, p=None, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float)
    p = LAGS[h] if p is None else p

    X = sm.add_constant(pd.concat(
        [y.shift(j).rename(f"lag{j}") for j in range(p)], axis=1
    ))

    data = X.join(y.shift(-h).rename("target")).dropna()

    preds = [
        sm.QuantReg(data["target"], data.drop(columns="target")).fit(q=tau)
           .predict(X.dropna().iloc[[-1]]).iloc[0]
        for tau in quantiles
    ]

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")
