"""Shared machinery for the benchmark forecasters.

All models here follow the same contract:

    model.fit(y)                     -> self
    model.predict_quantiles(h_max)   -> (h_max, n_quantiles) array

so a foundation model can be dropped in behind the same interface.

Two conventions are fixed across every benchmark:

*Direct* multi-horizon.
    Each horizon h gets its own estimated model, mapping the information set at
    the forecast origin straight to y_{t+h}. This matches how a time-series
    foundation model emits h = 1..H in one shot, and it avoids the understated
    intervals of a recursive scheme that ignores accumulated innovation
    uncertainty. The cost is a smaller effective sample as h grows, and
    regression errors that are serially correlated by construction.

*Quantiles* as the output format.
    The predictive distribution is represented by its quantiles on a fixed grid.
    This is what foundation models emit, it is what the pinball loss and the
    quantile approximation of CRPS consume, and it keeps parametric and
    distribution-free benchmarks directly comparable.
"""

from abc import ABC, abstractmethod

import numpy as np


# A 19-point grid. Dense enough that the averaged pinball loss is a decent
# approximation to CRPS, and it contains the 0.1..0.9 decile grid that Chronos
# and TimesFM emit by default.
DEFAULT_QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)


class BaseForecaster(ABC):
    """Interface shared by every benchmark model."""

    def __init__(self, quantiles=None):

        self.quantiles = np.asarray(
            DEFAULT_QUANTILES if quantiles is None else quantiles,
            dtype=float
        )

        if np.any(np.diff(self.quantiles) <= 0):
            raise ValueError("quantiles must be strictly increasing")

        if self.quantiles[0] <= 0 or self.quantiles[-1] >= 1:
            raise ValueError("quantiles must lie strictly inside (0, 1)")

        self.y_ = None

    @abstractmethod
    def fit(self, y):
        """Estimate on a 1-d series. NaNs are permitted inside the sample."""

    @abstractmethod
    def predict_quantiles(self, h_max, history=None):
        """Return predictive quantiles, shape (h_max, len(self.quantiles)).

        `history`, when given, must extend the series passed to fit(). The
        estimated model is applied to that later forecast origin without
        re-estimating, which is what makes a rolling-origin backtest with
        periodic re-estimation affordable.
        """

    def _store(self, y):

        y = np.asarray(y, dtype=float).ravel()

        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")

        self.y_ = y

        return y

    def _check_fitted(self):

        if self.y_ is None:
            raise RuntimeError("call fit() before predict_quantiles()")


def direct_design(y, p, h, seasonal_period=None):
    """Build the design for the direct h-step regression y_{t+h} ~ lags of y.

    Rows are the origins t with a complete set of p lags and an observed
    target; rows containing NaN are dropped (listwise). Returns

        X : (n, k) regressors - constant, p lags, optional seasonal dummies
        z : (n,)   targets y_{t+h}

    The lag block is ordered [y_t, y_{t-1}, ..., y_{t-p+1}].
    """

    n_obs = len(y)
    origins = np.arange(p - 1, n_obs - h)

    if len(origins) == 0:
        raise ValueError(
            f"not enough observations: need > {p + h - 1}, have {n_obs}"
        )

    lags = np.column_stack([
        y[origins - j] for j in range(p)
    ])

    targets = y[origins + h]

    blocks = [np.ones((len(origins), 1)), lags]

    if seasonal_period:
        blocks.append(
            seasonal_dummies(origins + h, seasonal_period)
        )

    X = np.hstack(blocks)

    keep = np.isfinite(X).all(axis=1) & np.isfinite(targets)

    return X[keep], targets[keep]


def prediction_row(y, p, h, seasonal_period=None):
    """The single regressor row for forecasting from the end of the sample."""

    if len(y) < p:
        raise ValueError(f"need at least {p} observations to form the lags")

    origin = len(y) - 1
    lags = np.array([y[origin - j] for j in range(p)])

    if not np.isfinite(lags).all():
        missing = [origin - j for j in range(p) if not np.isfinite(y[origin - j])]
        raise ValueError(
            f"the {p} lags at the forecast origin contain NaN at positions "
            f"{missing}. Interpolate, shorten the lag order, or skip this "
            "origin - the model will not silently impute."
        )

    blocks = [np.array([1.0]), lags]

    if seasonal_period:
        blocks.append(
            seasonal_dummies(np.array([origin + h]), seasonal_period)[0]
        )

    return np.concatenate(blocks)


def seasonal_dummies(positions, period):
    """Deterministic seasonal dummies, first season dropped as the reference.

    Season is taken from the position in the series, so position 0 defines the
    reference season. The panels here start in January, so position % 12 is
    month - 1.
    """

    season = np.asarray(positions) % period
    dummies = np.zeros((len(season), period - 1))

    for k in range(1, period):
        dummies[:, k - 1] = (season == k).astype(float)

    return dummies


def rearrange(quantile_matrix):
    """Enforce monotone quantiles by sorting along the quantile axis.

    This is the rearrangement of Chernozhukov, Fernandez-Val & Galichon (2010):
    sorting a non-monotone quantile curve weakly improves estimation error, so
    it is a free fix for the crossing that separate per-quantile fits produce.
    """

    return np.sort(np.asarray(quantile_matrix, dtype=float), axis=-1)
