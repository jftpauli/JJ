"""Quantile AR: predictive quantiles of y_{t+h} from p lags of y.

One quantile regression per level tau (Koenker & Bassett 1978):

    Q(y_{t+h} | y_t, ..., y_{t-p+1}) = a + b_1 y_t + ... + b_p y_{t-p+1}

Fit on a stationary series (year-on-year inflation, growth rates), not levels.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import QUANTILES

# Largest lag order considered. For monthly year-on-year data 13 is the natural
# ceiling: since y_{t+1} - y_t = d_{t+1} - d_{t-11}, the thirteenth lag carries
# the base effect dropping out of the year-on-year window, and the criterion
# picks it at h = 1. At long horizons that term leaves the conditioning set and
# the criterion falls back to one or a few lags.
MAX_LAGS = 13


def select_lags(y, h, max_lags=MAX_LAGS):
    """Lag order minimising the Schwarz criterion of the median regression.

    Koenker & Machado (1999): SIC(p) = log(mean |u|) + (p + 1) log(n) / (2n),
    with u the residuals of the direct h-step LAD regression. Same loss as the
    model itself and robust to outliers, unlike an OLS-based BIC. All orders
    are fit on the same sample, the one the largest order leaves, so their
    criteria are comparable.
    """

    lags = pd.concat([y.shift(j) for j in range(max_lags)], axis=1)
    data = lags.assign(target=y.shift(-h)).dropna().to_numpy()
    n = len(data)
    target = data[:, -1]

    best_sic, best_p = np.inf, 1
    for p in range(1, max_lags + 1):
        X = sm.add_constant(data[:, :p], has_constant="add")
        resid = sm.QuantReg(target, X).fit(q=0.5, max_iter=5000).resid
        sic = np.log(np.mean(np.abs(resid))) + (p + 1) * np.log(n) / (2 * n)
        if sic < best_sic:
            best_sic, best_p = sic, p

    return best_p


def quantile_ar(y, h=12, p=None, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample.

    p is chosen by the median-regression SIC (select_lags) unless given.
    """

    y = pd.Series(y).astype(float)
    p = select_lags(y, h) if p is None else p

    X = sm.add_constant(pd.concat(
        [y.shift(j).rename(f"lag{j}") for j in range(p)], axis=1
    ))

    data = X.join(y.shift(-h).rename("target")).dropna()

    preds = [
        sm.QuantReg(data["target"], data.drop(columns="target")).fit(q=tau, max_iter=5000)
           .predict(X.dropna().iloc[[-1]]).iloc[0]
        for tau in quantiles
    ]

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")
