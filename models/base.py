"""Shared machinery for the benchmark forecasters.

All models here follow the same contract:

    model.fit(y)                     -> self
    model.predict_quantiles(h_max)   -> (h_max, n_quantiles) array

so a foundation model can be dropped in behind the same interface.

A model that conditions on more than the target's own past sets the class
attribute `uses_exog = True` and takes two extra keyword arguments:

    model.fit(y, exog=X)
    model.predict_quantiles(h_max, history=..., exog_history=...)

where `X` has shape (len(y), k) and row i is the covariate vector observed at
the same period as y[i]. Callers switch on `uses_exog` rather than passing the
covariates blindly, so the univariate models keep their two-argument signature
and nothing has to accept an argument it would ignore.

Covariates enter *contemporaneously with the forecast origin only*. A model
here never sees a covariate value dated after the origin, so no assumption
about the future path of the conditioning variable is smuggled in - the
growth-at-risk regressions this supports (Adrian, Boyarchenko & Giannone 2019;
Loria, Matthes & Zhang 2025) are specified exactly that way.

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

    # Set True by models that take `exog` / `exog_history`. See the module
    # docstring: callers dispatch on this rather than on isinstance checks.
    uses_exog = False

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


def direct_design(y, p, h, seasonal_period=None, exog=None):
    """Build the design for the direct h-step regression y_{t+h} ~ lags of y.

    Rows are the origins t with a complete set of p lags and an observed
    target; rows containing NaN are dropped (listwise). Returns

        X : (n, k) regressors - constant, p lags, optional seasonal dummies,
                   optional covariates dated at the origin
        z : (n,)   targets y_{t+h}

    The lag block is ordered [y_t, y_{t-1}, ..., y_{t-p+1}].

    `exog`, when given, is (len(y), k) with row i observed at the same period
    as y[i]; row t of the design takes exog[t] - the value at the origin, never
    at the target date. A covariate row containing NaN drops that origin, the
    same listwise rule the lags follow.
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

    if exog is not None:
        blocks.append(check_exog(exog, n_obs)[origins])

    X = np.hstack(blocks)

    keep = np.isfinite(X).all(axis=1) & np.isfinite(targets)

    return X[keep], targets[keep]


def prediction_row(y, p, h, seasonal_period=None, exog=None):
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

    if exog is not None:

        row = check_exog(exog, len(y))[origin]

        if not np.isfinite(row).all():
            raise ValueError(
                "the covariates at the forecast origin contain NaN. Skip this "
                "origin - the model will not silently impute."
            )

        blocks.append(row)

    return np.concatenate(blocks)


def check_exog(exog, n_obs):
    """Validate a covariate block and return it as a 2-d float array."""

    exog = np.asarray(exog, dtype=float)

    if exog.ndim == 1:
        exog = exog[:, None]

    if exog.ndim != 2:
        raise ValueError("exog must be 1- or 2-dimensional")

    if len(exog) != n_obs:
        raise ValueError(
            f"exog has {len(exog)} rows but the series has {n_obs} - the two "
            "must be aligned period by period"
        )

    return exog


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
