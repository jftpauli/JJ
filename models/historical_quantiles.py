"""Unconditional historical quantiles - the climatological baseline.

The predictive distribution is simply the empirical distribution of the series'
own past values, identical at every horizon:

    F_hat(y_{T+h}) = empirical CDF of {y_1, ..., y_T}

It conditions on nothing. That is the point: it is the reference a forecast has
to beat before it can be said to contain any information at all, and it is the
standard reference for a CRPS skill score,

    CRPSS = 1 - CRPS_model / CRPS_climatology

so a model scoring 0 has no skill over knowing the historical distribution, and
a negative score is actively worse than that.

Because it is horizon-invariant, its CRPS is flat in h by construction. A
conditional model should beat it clearly at short horizons and converge toward
it as h grows - if a model never beats it, that model has learned nothing about
the dynamics; if it converges to it quickly, the series is unforecastable
beyond that horizon.

Fit on the same stationary transform used for the other models. Run on an index
level the "historical distribution" is a mix of every past regime and price
level, which is not a meaningful predictive distribution.
"""

import numpy as np

from .base import BaseForecaster, rearrange


class HistoricalQuantiles(BaseForecaster):
    """Empirical quantiles of the series' own history.

    Parameters
    ----------
    window : int or None
        Number of most recent observations to use. None uses the full history
        available at the forecast origin (an expanding window). A finite window
        tracks slow shifts in the unconditional distribution - a 20-year window
        keeps 1990s inflation levels out of a 2020s forecast - at the cost of a
        noisier estimate.
    seasonal_period : int or None
        If set, the quantiles are taken over past observations sharing the
        target's season (a seasonal climatology). Leave None for seasonally
        adjusted or year-on-year data, where seasonality is already removed.
    min_observations : int
        Refuse to form quantiles from fewer than this many values.
    """

    def __init__(self, window=None, seasonal_period=None, quantiles=None,
                 min_observations=20):

        super().__init__(quantiles=quantiles)

        if window is not None and window < min_observations:
            raise ValueError("window must be at least min_observations")

        self.window = window
        self.seasonal_period = seasonal_period
        self.min_observations = int(min_observations)

    def fit(self, y):

        y = self._store(y)

        if np.isfinite(y).sum() < self.min_observations:
            raise ValueError(
                f"only {int(np.isfinite(y).sum())} finite observations; "
                f"need {self.min_observations}"
            )

        return self

    def predict_quantiles(self, h_max, history=None):
        """Predictive quantiles, identical across all horizons.

        Unlike the estimated benchmarks, this one *does* recompute from
        `history` when given. There is no estimation cost to amortise, so the
        honest definition is the distribution of everything observed up to the
        forecast origin rather than a stale one carried over from the last
        re-estimation.
        """

        self._check_fitted()

        y = self.y_ if history is None else np.asarray(history, dtype=float)

        out = np.empty((h_max, len(self.quantiles)))

        for i, h in enumerate(range(1, h_max + 1)):
            out[i] = self._empirical_quantiles(y, h)

        return rearrange(out)

    def _empirical_quantiles(self, y, h):
        """Quantiles of the relevant slice of history for horizon h."""

        if self.seasonal_period:
            # Observations sharing the season of the target period T + h.
            target_season = (len(y) - 1 + h) % self.seasonal_period
            positions = np.arange(len(y))
            values = y[positions % self.seasonal_period == target_season]
        else:
            values = y

        values = values[np.isfinite(values)]

        if self.window is not None:
            values = values[-self.window:]

        if len(values) < self.min_observations:
            raise ValueError(
                f"only {len(values)} usable observations for horizon {h}; "
                f"need {self.min_observations}"
            )

        return np.quantile(values, self.quantiles)
