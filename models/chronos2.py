"""Chronos-2 - the pretrained time-series foundation model, zero-shot.

Amazon's Chronos-2 (120M parameters, encoder-decoder over patched values). It is
used here exactly as it ships: no fine-tuning and no per-series adaptation. At
each forecast origin the model sees the series' own past - and, when the target
declares covariates, their past as well - and emits the whole predictive
distribution for h = 1..H in one forward pass.

That is the comparison worth running. The point of a foundation model is that it
brings information from *other* series - it has seen macro data, and inflation
in particular, during pre-training - so it should be able to price the tails of
a country's inflation distribution better than a quantile regression that has
only ~400 monthly observations of that one country to learn from. Whether it
actually does is what the backtest answers.

Fits the same contract as the estimated benchmarks:

    model.fit(y).predict_quantiles(h_max, history=...)  ->  (h_max, n_q)

though `fit` does nothing but record the series. Chronos-2 has no parameters to
estimate, so the backtest's re-estimation schedule is irrelevant to it; every
origin gets a fresh forward pass over the history up to that origin. This is
also why it is *not* advantaged by the periodic-refit scheme the estimated
benchmarks run under - if anything the reverse, since QuantileAR's coefficients
can be up to `refit_every` periods stale while these forecasts never are.

Three things to state in any write-up that uses this:

* **Pre-training leakage is not controlled.** Chronos-2 was trained on public
  corpora with an unknown relationship to OECD CPI, and its training window
  overlaps the backtest's evaluation period. A rolling-origin backtest keeps
  *this* code from peeking ahead, but it cannot un-see whatever the weights
  already encode. The scores here are therefore an upper bound on what a
  genuinely out-of-sample foundation model would deliver, and the estimated
  benchmarks - refit only on data available at each origin - are strictly
  honest in a way this model is not. Do not present the gap as if the two were
  on equal footing.
* **The direct/recursive distinction still applies.** h = 1..12 comes out of a
  single forward pass because 12 <= the model's 16-step output patch, so no
  autoregressive unrolling happens and no error accumulates through a feedback
  loop. That matches the direct-multi-horizon convention the other benchmarks
  follow (see base.py).
* **Two of the 23 quantile levels are interpolated.** The model's native grid is
  [0.01, 0.05, 0.1, 0.15, ..., 0.95, 0.99]; the 0.025 and 0.975 levels this
  project scores on are not on it, and the pipeline fills them by linear
  interpolation between neighbouring levels. That flattens the shape of the
  predictive density between 1% and 5% (and its mirror), so read the far tails
  of the PIT histogram with that in mind. Every other level is native.

Deterministic: Chronos-2 emits quantiles from a quantile head rather than
sampling paths, so repeated calls on the same history return identical numbers
and no seed is involved.
"""

import numpy as np

from .base import BaseForecaster, check_exog, rearrange


DEFAULT_MODEL_ID = "amazon/chronos-2"

# The model's own quantile grid, from its config. Levels outside it are
# extrapolated by the pipeline and levels between its points are interpolated;
# this is here so callers can check which of their levels are native.
NATIVE_QUANTILES = np.array([0.01] + list(np.round(np.arange(0.05, 0.96, 0.05), 2)) + [0.99])

# Loading the weights costs a couple of seconds and ~500MB. The backtest builds
# a fresh model object at every re-estimation, so without this the same weights
# would be read off disk dozens of times per run.
_PIPELINES = {}


def load_pipeline(model_id=DEFAULT_MODEL_ID, device="auto"):
    """The Chronos pipeline for `model_id`, loaded once per process.

    `device="auto"` uses a GPU when one is visible and falls back to CPU. On CPU
    a 12-step forecast for four series takes under a tenth of a second, so the
    whole rolling-origin backtest is a matter of seconds - there is no reason to
    require a GPU for this.
    """

    import torch
    from chronos import BaseChronosPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = (model_id, device)

    if key not in _PIPELINES:
        _PIPELINES[key] = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device
        )

    return _PIPELINES[key]


class Chronos2(BaseForecaster):
    """Zero-shot Chronos-2, behind the benchmark forecaster interface.

    Parameters
    ----------
    model_id : str
        Hugging Face identifier. Weights are resolved through the usual HF
        cache, so an already-downloaded model works offline.
    device : str
        "auto", "cpu", "cuda", or any device string torch accepts.
    context_length : int or None
        Most recent observations shown to the model. None uses the model's own
        maximum (8192 for Chronos-2), which is far longer than any series here,
        so in practice the model sees the full history at each origin - the
        same information set the climatological and quantile-AR benchmarks get.
    min_observations : int
        Refuse to forecast from fewer finite observations than this. The model
        will happily produce something from a handful of points; scoring that
        against benchmarks that require a real sample is not a fair comparison.
    uses_exog : bool
        Whether to pass covariates to the model. They go in as
        `past_covariates` - values up to and including the forecast origin,
        nothing beyond it. Chronos-2 also accepts `future_covariates`, which
        would require knowing the covariate's path over the forecast window;
        that is not information a forecaster has, and it is not what the
        growth-at-risk regressions condition on either.
    covariate_names : sequence of str or None
        Names for the covariate columns. Only used to key the dict the pipeline
        expects; the model treats them as opaque. Defaults to x1, x2, ...
    """

    def __init__(self, model_id=DEFAULT_MODEL_ID, device="auto",
                 context_length=None, quantiles=None, min_observations=20,
                 uses_exog=False, covariate_names=None):

        super().__init__(quantiles=quantiles)

        self.model_id = model_id
        self.device = device
        self.context_length = context_length
        self.min_observations = int(min_observations)
        self.uses_exog = bool(uses_exog)
        self.covariate_names = covariate_names

        self.exog_ = None

    def fit(self, y, exog=None):
        """Record the series. There is nothing to estimate.

        The pipeline is loaded here rather than lazily inside predict, so a
        missing or unreadable checkpoint fails at the same point in the backtest
        where an estimation failure would - not halfway through scoring.
        """

        y = self._store(y)

        if np.isfinite(y).sum() < self.min_observations:
            raise ValueError(
                f"only {int(np.isfinite(y).sum())} finite observations; "
                f"need {self.min_observations}"
            )

        if self.uses_exog:

            if exog is None:
                raise ValueError(
                    "this model was constructed with uses_exog=True but fit() "
                    "was given no covariates"
                )

            self.exog_ = check_exog(exog, len(y))

        elif exog is not None:
            raise ValueError(
                "covariates passed to a model constructed with uses_exog=False"
            )

        load_pipeline(self.model_id, self.device)

        return self

    def _covariate_dict(self, exog):
        """The `past_covariates` mapping the pipeline expects."""

        names = self.covariate_names

        if names is None:
            names = [f"x{j + 1}" for j in range(exog.shape[1])]

        if len(names) != exog.shape[1]:
            raise ValueError(
                f"{len(names)} covariate names for {exog.shape[1]} columns"
            )

        return {name: exog[:, j] for j, name in enumerate(names)}

    def predict_quantiles(self, h_max, history=None, exog_history=None):
        """Predictive quantiles for h = 1..h_max from one forward pass.

        Like HistoricalQuantiles, this recomputes from `history` when given -
        there is no fitted state to carry forward, so the honest definition is
        always "everything observed up to this origin".

        NaNs inside the context are passed through rather than imputed or
        rejected: Chronos-2 handles missing values natively, which the estimated
        benchmarks cannot. That means this model can produce a forecast at an
        origin where QuantileAR refuses (a NaN among its lags), so the two are
        not always scored on identical origin sets. The per-horizon tables carry
        `n_forecasts` for exactly this reason, and the paired comparison in
        run_benchmarks.py intersects on common origins before differencing.
        """

        self._check_fitted()

        y = self.y_ if history is None else np.asarray(history, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if np.isfinite(y).sum() < self.min_observations:
            raise ValueError(
                f"only {int(np.isfinite(y).sum())} finite observations at the "
                f"forecast origin; need {self.min_observations}"
            )

        pipeline = load_pipeline(self.model_id, self.device)

        if self.uses_exog:
            exog = self.exog_ if exog_history is None else exog_history
            exog = check_exog(exog, len(y))
            inputs = [{
                "target": y,
                "past_covariates": self._covariate_dict(exog)
            }]
        elif exog_history is not None:
            raise ValueError(
                "covariates passed to a model constructed with uses_exog=False"
            )
        else:
            inputs = [y]

        quantiles, _ = pipeline.predict_quantiles(
            inputs,
            prediction_length=int(h_max),
            quantile_levels=[float(tau) for tau in self.quantiles],
            context_length=self.context_length,
            limit_prediction_length=False
        )

        # (n_series=1, n_variates=1, h_max, n_quantiles) -> (h_max, n_quantiles)
        out = quantiles[0][0].to("cpu").numpy().astype(float)

        if out.shape != (int(h_max), len(self.quantiles)):
            raise RuntimeError(
                f"chronos returned {out.shape}, expected "
                f"{(int(h_max), len(self.quantiles))}"
            )

        # Interpolated levels can in principle come back marginally out of
        # order; sorting is free and keeps the scoring code's monotonicity
        # assumption safe.
        return rearrange(out)
