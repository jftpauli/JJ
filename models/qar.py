"""Quantile AR: predictive quantiles of y_{t+h} from p lags of y.

One quantile regression per level tau (Koenker & Bassett 1978):

    Q(y_{t+h} | y_t, ..., y_{t-p+1}) = a + b_1 y_t + ... + b_p y_{t-p+1}

Fit on a stationary series (year-on-year inflation, growth rates), not levels.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def select_lags(y, h=12, p_max=12, quantiles=QUANTILES, penalty="bic"):
    """Lag order by the quantile-regression information criterion.

    The check function is an asymmetric Laplace log-likelihood with the scale
    profiled out (Machado 1993), giving 2n*log(mean rho) + penalty*k. Every
    candidate is fit on the sample available at p_max, or the criteria would
    not be comparable, and the criterion is summed over the quantile grid so
    one lag order serves every tau.
    """

    y = pd.Series(y).astype(float)

    lags = pd.concat([y.shift(j).rename(f"lag{j}") for j in range(p_max)], axis=1)
    data = sm.add_constant(lags).join(y.shift(-h).rename("target")).dropna()

    z, n = data["target"], len(data)
    k = np.log(n) if penalty == "bic" else 2.0

    def ic(p):
        X = data[["const"] + [f"lag{j}" for j in range(p)]]
        resid = [z - sm.QuantReg(z, X).fit(q=tau).predict(X) for tau in quantiles]
        return sum(
            2 * n * np.log(np.mean(r * (tau - (r < 0)))) + k * X.shape[1]
            for tau, r in zip(quantiles, resid)
        )

    return pd.Series({p: ic(p) for p in range(1, p_max + 1)}, name=penalty)


def quantile_ar(y, h=12, p=1, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample."""

    y = pd.Series(y).astype(float)

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
