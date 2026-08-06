"""Univariate probabilistic benchmark models.

Every model exposes the same two calls,

    model.fit(y).predict_quantiles(h_max)  ->  (h_max, n_quantiles) array

so a time-series foundation model can be evaluated behind the same interface.

    HistoricalQuantiles  unconditional empirical quantiles - the climatological
                         reference a forecast must beat to show any skill
    QuantileAR           direct quantile AR - distribution-free and conditional

See base.py for the direct-multi-horizon and quantile-output conventions, and
scoring.py for the pinball / CRPS / coverage utilities used to compare them.
"""

from .base import DEFAULT_QUANTILES, BaseForecaster, rearrange
from .historical_quantiles import HistoricalQuantiles
from .qar import QuantileAR
from .scoring import (
    coverage,
    crps_from_quantiles,
    crps_skill_score,
    pinball_loss,
    pit_from_quantiles,
    predictive_mean,
)

__all__ = [
    "BaseForecaster",
    "DEFAULT_QUANTILES",
    "HistoricalQuantiles",
    "QuantileAR",
    "coverage",
    "crps_from_quantiles",
    "crps_skill_score",
    "pinball_loss",
    "pit_from_quantiles",
    "predictive_mean",
    "rearrange",
]
