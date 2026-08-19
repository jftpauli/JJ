"""Univariate probabilistic benchmark models.

Every model exposes the same two calls,

    model.fit(y).predict_quantiles(h_max)  ->  (h_max, n_quantiles) array

so a time-series foundation model can be evaluated behind the same interface.

    HistoricalQuantiles  unconditional empirical quantiles - the climatological
                         reference a forecast must beat to show any skill
    QuantileAR           direct quantile AR - distribution-free and conditional
    Chronos2             pretrained foundation model, zero-shot - the thing the
                         two benchmarks above exist to be measured against

See base.py for the direct-multi-horizon and quantile-output conventions, and
scoring.py for the pinball / CRPS / coverage utilities used to compare them.
"""

from .base import DEFAULT_QUANTILES, BaseForecaster, rearrange
from .chronos2 import Chronos2
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
from .sundial import Sundial

__all__ = [
    "BaseForecaster",
    "Chronos2",
    "DEFAULT_QUANTILES",
    "HistoricalQuantiles",
    "QuantileAR",
    "Sundial",
    "coverage",
    "crps_from_quantiles",
    "crps_skill_score",
    "pinball_loss",
    "pit_from_quantiles",
    "predictive_mean",
    "rearrange",
]
