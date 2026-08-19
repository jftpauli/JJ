"""Sundial - generative time-series foundation model, zero-shot.

This wrapper follows the same interface as Chronos-2, but Sundial is
sample-based rather than quantile-native. The project still needs a matrix of
predictive quantiles, so each forecast samples multiple future paths and then
computes the empirical quantiles on the grid used downstream.

The public contract remains:

    model.fit(y).predict_quantiles(h_max, history=...) -> (h_max, n_q)

so the rest of the repository can score it exactly like any other benchmark.
"""

from __future__ import annotations

import numpy as np

from .base import BaseForecaster, rearrange


DEFAULT_MODEL_ID = "thuml/sundial-base-128m"

# The project expects a single, fixed quantile grid across all models; the
# generative model emits samples and we convert them manually.
# This mirrors the way run_benchmarks.py arranges its comparisons.
NATIVE_QUANTILES = None

_PIPELINES = {}


def _normalise_sample_array(samples, h_max):
    """Convert model output to an array shaped (n_samples, h_max)."""

    arr = np.asarray(samples, dtype=float)

    # Common generator shapes are (n_samples, h_max) or (1, n_samples, h_max)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim == 2:
        if arr.shape[1] != int(h_max):
            if arr.shape[0] == int(h_max):
                arr = arr.T
            else:
                raise ValueError(
                    f"unexpected sample shape {arr.shape}; expected ({h_max}, n_samples) or (n_samples, {h_max})"
                )
        return arr

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[1] == 1:
            return arr[:, 0, :]

    raise ValueError(
        f"unsupported sample tensor shape {arr.shape}; expected (n_samples, h_max)"
    )


def load_pipeline(model_id=DEFAULT_MODEL_ID, device="auto"):
    """Load a Sundial model, reusing it within one Python process.

    Sundial ships no dedicated pip package; the model card loads it as custom
    `transformers` code:

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    `trust_remote_code=True` runs code from the checkpoint repo, so `model_id`
    must be a source you trust (the default points at the official thuml repo).
    """

    import torch
    from transformers import AutoModelForCausalLM
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = (model_id, device)

    if key in _PIPELINES:
        return _PIPELINES[key]

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True
    ).to(device)
    model.eval()

    _PIPELINES[key] = model
    return model


class Sundial(BaseForecaster):
    """Generative Sundial wrapper behind the benchmark forecaster interface.

    Parameters
    ----------
    model_id : str
        Model identifier. The default is the current Sundial checkpoint name used
        by the project, but the wrapper is intentionally flexible and accepts any
        local or HuggingFace checkpoint exposing a generative forecasting API.
    device : str
        Device passed to the underlying model.
    context_length : int or None
        Maximum number of recent observations shown to the model. None keeps the
        model default, which is usually large enough for the series here.
    min_observations : int
        Minimum finite observations required before producing a forecast.
    num_samples : int
        Number of future draws used to approximate predictive quantiles.
    """

    def __init__(self, model_id=DEFAULT_MODEL_ID, device="auto",
                 context_length=None, quantiles=None, min_observations=20,
                 num_samples=10):

        super().__init__(quantiles=quantiles)

        self.model_id = model_id
        self.device = device
        self.context_length = context_length
        self.min_observations = int(min_observations)
        self.num_samples = int(num_samples)

    def fit(self, y):
        """Record the series. There is no parameter estimation to do."""

        y = self._store(y)

        if np.isfinite(y).sum() < self.min_observations:
            raise ValueError(
                f"only {int(np.isfinite(y).sum())} finite observations; "
                f"need {self.min_observations}"
            )

        load_pipeline(self.model_id, self.device)
        return self

    def _generate_samples(self, y, h_max):
        """Generate genuinely stochastic Sundial forecast paths."""

        import torch

        model = load_pipeline(self.model_id, self.device)
        device = next(model.parameters()).device

        context = (
            y
            if self.context_length is None
            else y[-int(self.context_length):]
        )

        # Copy the array so PyTorch does not receive a non-writable NumPy view.
        context = np.array(context, dtype=np.float32, copy=True)

        seqs = torch.as_tensor(
            context,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():
            output = model.generate(
                seqs,
                max_new_tokens=int(h_max),
                num_samples=int(self.num_samples),
            )

        samples = output.detach().cpu().numpy()

        print(
            f"      Sundial raw output shape: {samples.shape}",
            flush=True
        )

        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples[0]

        samples = _normalise_sample_array(samples, h_max)

        print(
            f"      Sundial samples: shape={samples.shape}, "
            f"unique={np.unique(samples, axis=0).shape[0]}, "
            f"std={samples.std():.6f}",
            flush=True
        )

        if samples.shape[0] < 2:
            raise RuntimeError(
                f"Sundial returned only {samples.shape[0]} sample path(s); "
                f"expected {self.num_samples}."
            )

        if np.allclose(samples, samples[0:1]):
            raise RuntimeError(
                "Sundial generated identical sample paths. "
                "Probabilistic generation is not working."
            )

        return samples

    def predict_quantiles(self, h_max, history=None):
        """Predictive quantiles for h = 1..h_max from Monte Carlo samples."""

        self._check_fitted()

        y = self.y_ if history is None else np.asarray(history, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if np.isfinite(y).sum() < self.min_observations:
            raise ValueError(
                f"only {int(np.isfinite(y).sum())} finite observations at the "
                f"forecast origin; need {self.min_observations}"
            )

        h_max = int(h_max)
        samples = self._generate_samples(y, h_max)

        if samples.shape[1] != h_max:
            raise RuntimeError(
                f"Sundial samples had shape {samples.shape}, expected (n_samples, {h_max})"
            )

        # Empirical quantiles of the simulated future paths. This is the right
        # translation from a generative model to the quantile format the rest of
        # the project consumes: CRPS / PIT / coverage all work on quantiles.
        out = np.quantile(samples, self.quantiles, axis=0).T

        if out.shape != (h_max, len(self.quantiles)):
            raise RuntimeError(
                f"Sundial produced {out.shape}, expected {(h_max, len(self.quantiles))}"
            )

        return rearrange(out)
