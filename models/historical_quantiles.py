"""Unconditional historical quantiles - the climatological baseline.

The predictive distribution is the empirical distribution of the series' own
past values, identical at every horizon:

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
import pandas as pd

from config import QUANTILES


def historical_quantiles(y, h=12, quantiles=QUANTILES):
    """Forecast the quantiles of y_{t+h} from the end of the sample.

    `h` is accepted to match the other forecasters and ignored: the answer is
    the same at every horizon. Pass the history up to the origin - there is no
    estimation to amortise, so a rolling-origin backtest just calls this again.
    """

    y = pd.Series(y).astype(float).dropna()

    return pd.Series(np.quantile(y, quantiles), index=quantiles, name="forecast")
