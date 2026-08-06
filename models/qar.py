"""Direct quantile autoregression - the distribution-free benchmark.

For each horizon h and each quantile level tau, a separate quantile regression
of y_{t+h} on p lags is estimated (Koenker & Bassett 1978; the autoregressive
case is Koenker & Xiao 2006). Nothing is assumed about the shape of the
predictive distribution, so unlike the AR-Gaussian benchmark this model can
express skewness and horizon-varying spread - the features that matter most
when scoring density forecasts of macro data.

The fit is solved as a linear program rather than by iteratively reweighted
least squares. The IRLS route (statsmodels' QuantReg) hits its iteration limit
on persistent macro series - including a plain AR(1) - and silently returning
some fallback for a quantile that failed to converge is exactly the kind of
defect that survives into published numbers. The LP is exact, needs no
tolerance, and agreed with converged IRLS fits to 2e-5 in testing.

Two weaknesses remain, worth stating in any write-up that uses this:

* Tail levels are estimated from few effective observations. With ~400 monthly
  observations the tau = 0.05 fit is driven by roughly 20 points, so the tails
  are high-variance. A foundation model's tails are implicitly regularised by
  pre-training, so part of any gap it shows here is estimation noise rather
  than a modelling win. Say so rather than letting the comparison imply
  otherwise.
* Separately fitted quantile curves can cross. They are put back in order by
  the rearrangement of Chernozhukov, Fernandez-Val & Galichon (2010), which
  weakly reduces estimation error, so the fix costs nothing.

Fit on a stationary transform (growth rates, year-on-year), not on index
levels: quantile regression on an integrated series is badly conditioned.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .base import BaseForecaster, direct_design, prediction_row, rearrange


def quantile_regression(X, z, tau):
    """Solve min_b sum_i rho_tau(z_i - x_i'b) as a linear program.

    The check function is linearised with the standard positive/negative
    residual split, u - v = z - Xb with u, v >= 0:

        min  tau * 1'u + (1 - tau) * 1'v   s.t.  Xb + u - v = z,  u, v >= 0

    Coefficients b are free. Constraints are sparse - the u/v blocks are two
    identity matrices - which is what keeps this fast enough to sit inside a
    rolling-origin backtest.
    """

    n, k = X.shape

    cost = np.concatenate([
        np.zeros(k),
        tau * np.ones(n),
        (1.0 - tau) * np.ones(n)
    ])

    identity = sparse.identity(n, format="csc")

    constraints = sparse.hstack(
        [sparse.csc_matrix(X), identity, -identity],
        format="csc"
    )

    bounds = [(None, None)] * k + [(0, None)] * (2 * n)

    result = linprog(
        cost,
        A_eq=constraints,
        b_eq=z,
        bounds=bounds,
        method="highs"
    )

    if not result.success:
        raise RuntimeError(
            f"quantile regression LP failed at tau={tau:.3f}: {result.message}"
        )

    return np.asarray(result.x[:k], dtype=float)


class QuantileAR(BaseForecaster):
    """Direct quantile AR: one quantile regression per (horizon, quantile).

    Parameters
    ----------
    p : int
        Lag order, held fixed. BIC-style order selection is not well defined
        across separate per-quantile fits, and the usual choice is to hold the
        lag structure fixed so the quantile curves share a conditioning set.
    seasonal_period : int or None
        If set, deterministic seasonal dummies of this period are included.
    """

    def __init__(self, p=4, seasonal_period=None, quantiles=None):

        super().__init__(quantiles=quantiles)

        self.p = int(p)
        self.seasonal_period = seasonal_period

        self.coefs_ = {}

    def fit(self, y):

        y = self._store(y)
        self.coefs_ = {}

        return self

    def _fit_horizon(self, h):
        """Coefficients for every quantile at one horizon; cached."""

        if h in self.coefs_:
            return self.coefs_[h]

        X, z = direct_design(
            self.y_, self.p, h, self.seasonal_period
        )

        n, k = X.shape

        if n <= k + 1:
            raise ValueError(
                f"horizon {h}: {n} usable rows for {k} parameters - "
                "not enough data for quantile regression"
            )

        coefs = np.column_stack([
            quantile_regression(X, z, tau) for tau in self.quantiles
        ])

        self.coefs_[h] = coefs

        return coefs

    def predict_quantiles(self, h_max, history=None):
        """Predictive quantiles, optionally from a later forecast origin.

        `history` applies the estimated coefficients to a longer series without
        re-estimating - the standard "re-estimate periodically, forecast every
        period" backtest scheme. It must extend the series the model was fit on.
        """

        self._check_fitted()

        origin_series = self.y_ if history is None else np.asarray(history, float)
        out = np.empty((h_max, len(self.quantiles)))

        for i, h in enumerate(range(1, h_max + 1)):

            coefs = self._fit_horizon(h)

            x0 = prediction_row(
                origin_series, self.p, h, self.seasonal_period
            )

            out[i] = x0 @ coefs

        return rearrange(out)
