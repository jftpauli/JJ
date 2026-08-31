"""Univariate probabilistic benchmark models.

All three forecasters share one call,

    forecast(y, h)  ->  Series of predictive quantiles for y_{t+h}

on the QUANTILES grid in config.toml, so they are interchangeable at an origin.

    historical_quantiles unconditional empirical quantiles - the climatological
                         reference a forecast must beat to show any skill
    quantile_ar          direct quantile AR - distribution-free and conditional
    chronos2             pretrained foundation model, zero-shot - the thing the
                         two benchmarks above exist to be measured against

Each is *direct*: horizon h gets its own mapping from the origin's information
set to y_{t+h}, never a recursive unrolling.
"""

from .chronos2 import chronos2
from .historical_quantiles import historical_quantiles
from .qar import quantile_ar

__all__ = ["chronos2", "historical_quantiles", "quantile_ar"]
