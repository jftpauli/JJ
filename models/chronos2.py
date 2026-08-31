"""Chronos-2 - the pretrained time-series foundation model, zero-shot.

Amazon's Chronos-2 (120M parameters) used exactly as it ships: no fine-tuning
and no per-series adaptation. At each origin it sees the series' own past and
emits the predictive distribution for h = 1..H in one forward pass, so h = 12
involves no autoregressive unrolling (12 <= its 16-step output patch) and the
direct-multi-horizon convention the other benchmarks follow still holds.

That is the comparison worth running: a foundation model brings information
from *other* series - it has seen macro data, and inflation in particular,
during pre-training - so it should price the tails better than a quantile
regression with only ~400 monthly observations of one country to learn from.

Two things to state in any write-up that uses this:

* **Pre-training leakage is not controlled.** Chronos-2 was trained on public
  corpora with an unknown relationship to OECD CPI, and its training window
  overlaps the evaluation period. A rolling-origin backtest keeps *this* code
  from peeking ahead, but it cannot un-see what the weights already encode. The
  scores are an upper bound on what a genuinely out-of-sample foundation model
  would deliver; the estimated benchmarks are honest in a way this is not.
* **Every level scored here is native.** The model's own grid is
  [0.01, 0.05, 0.10, ..., 0.95, 0.99], which is exactly the QUANTILES grid in
  config.toml - nothing is interpolated. Add a level off that grid and the
  pipeline will fill it by linear interpolation between neighbours, flattening
  the density there; the far tails are where that would bite.

Deterministic: quantiles come from a quantile head rather than sampled paths,
so repeated calls on the same history return identical numbers, no seed.
"""

import numpy as np
import pandas as pd

from config import QUANTILES

MODEL_ID = "amazon/chronos-2"

# Loading the weights costs a couple of seconds and ~500MB; a rolling-origin
# backtest calls this at every origin.
_PIPELINES = {}


def load_pipeline(model_id=MODEL_ID, device="auto"):
    """The Chronos pipeline for `model_id`, loaded once per process."""

    import torch
    from chronos import BaseChronosPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if (model_id, device) not in _PIPELINES:
        _PIPELINES[model_id, device] = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device
        )

    return _PIPELINES[model_id, device]


def chronos2(y, h=12, quantiles=QUANTILES, model_id=MODEL_ID, device="auto"):
    """Forecast the quantiles of y_{t+h} from the end of the sample.

    NaNs inside the history are passed through rather than imputed: Chronos-2
    handles missing values natively, which the estimated benchmarks cannot. It
    can therefore forecast at origins where quantile_ar refuses (a NaN among
    its lags), so intersect on common origins before scoring the two against
    each other.
    """

    y = pd.Series(y).astype(float)

    preds, _ = load_pipeline(model_id, device).predict_quantiles(
        [y.to_numpy()],
        prediction_length=int(h),
        quantile_levels=[float(tau) for tau in quantiles],
        limit_prediction_length=False
    )

    # (n_series=1, n_variates=1, h, n_quantiles) -> the row for horizon h
    row = preds[0][0].to("cpu").numpy().astype(float)[int(h) - 1]

    return pd.Series(np.sort(row), index=quantiles, name="forecast")
