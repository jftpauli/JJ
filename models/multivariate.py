"""Forecasters conditioned on a financial conditions index (FCI) as covariate.

Each function has the signature f(y, x, h) -> pd.Series of QUANTILES-indexed
predictive quantiles for y_{t+h}, where y is the target history up to the
origin t and x the FCI history on the same index (NaN where the FCI is not
available, e.g. before 1995). The FCI enters as a *past-only* covariate: no
model sees FCI values after the origin, so nothing has to be forecast for it
and nothing leaks.

* chronos2_x  - Chronos-2, FCI as `past_covariates` (group attention).
* t0gamma_x   - t0-gamma, FCI as a HISTORICAL variate in the target's group.
* timestral_x - Timestral, FCI as `past_covariates`.
* qar_x       - quantile AR augmented with lags of the FCI (the benchmark).

The foundation models are used zero-shot with the same settings as their
univariate wrappers in models/, so the only change is the extra input row.
"""

import numpy as np
import pandas as pd

from config import LOOKBACK, QUANTILES

# Largest number of FCI lags (x_t, ..., x_{t-k+1}) the QAR-X criterion considers.
MAX_X_LAGS = 3


def _aligned(y, x):
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float).reindex(y.index)
    return y, x


def chronos2_x(y, x, h=12, quantiles=QUANTILES):
    """Chronos-2 with the FCI as past-only covariate, full history as in chronos2()."""

    from models.chronos2 import load_pipeline

    y, x = _aligned(y, x)
    preds, _ = load_pipeline().predict_quantiles(
        [{"target": y.to_numpy(), "past_covariates": {"fci": x.to_numpy()}}],
        prediction_length=int(h),
        quantile_levels=[float(tau) for tau in quantiles],
        limit_prediction_length=False,
    )

    # (n_series=1, n_targets=1, h, n_quantiles) -> the row for horizon h
    row = preds[0][0].to("cpu").numpy().astype(float)[int(h) - 1]

    return pd.Series(np.sort(row), index=quantiles, name="forecast")


def t0gamma_x(y, x, h=12, lookback=LOOKBACK, quantiles=QUANTILES):
    """t0-gamma with the FCI as a HISTORICAL covariate row.

    The input is built as a t0 TimeSeries of width context + h: the target row
    and the covariate row share one group id, and both are WITHHELD over the
    horizon (the covariate is not known there).
    """

    import torch
    from t0.data import TimeSeries
    from t0.types import MaskType, VariateType

    from models.t0gamma import load_model

    y, x = _aligned(y, x)
    y = y.loc[y.first_valid_index():]
    x = x.reindex(y.index)
    values = np.vstack([y.to_numpy()[-lookback:], x.to_numpy()[-lookback:]]).astype(np.float32)

    model = load_model()
    device = next(model.parameters()).device
    n_rows, width, h = values.shape[0], values.shape[1], int(h)

    v = torch.as_tensor(values, device=device)
    is_nan = torch.isnan(v)
    mask = torch.where(is_nan, int(MaskType.MISSING), int(MaskType.VALID)).to(torch.int8)

    series = TimeSeries(
        variates=torch.cat([torch.nan_to_num(v, nan=0.0), torch.zeros(n_rows, h, device=device)], dim=1),
        mask=torch.cat(
            [mask, torch.full((n_rows, h), int(MaskType.WITHHELD), dtype=torch.int8, device=device)], dim=1
        ),
        group_ids=torch.zeros((n_rows, width + h), dtype=torch.int64, device=device),
        variate_type=torch.tensor(
            [[int(VariateType.TARGET)], [int(VariateType.HISTORICAL)]], dtype=torch.int64, device=device
        ).expand(n_rows, width + h).contiguous(),
    )

    forecast = model.predict(
        series, horizon=h, context_length=width, quantile_levels=[float(tau) for tau in quantiles]
    )

    # (target_rows=1, h, n_quantiles) -> the quantile row for horizon h
    row = forecast.quantiles[0, h - 1].to("cpu").numpy().astype(float)

    return pd.Series(np.sort(row), index=quantiles, name="forecast")


def timestral_x(y, x, h=12, lookback=LOOKBACK, quantiles=QUANTILES):
    """Timestral with the FCI as past-only covariate."""

    from models.timestral import load_model

    y, x = _aligned(y, x)
    y = y.loc[y.first_valid_index():]
    x = x.reindex(y.index)
    context = y.to_numpy()[-lookback:][None, :]
    past = x.to_numpy()[-lookback:][None, None, :]

    model = load_model()
    forecast = model.forecast(
        context, prediction_length=int(h), past_covariates=past, clip_nonnegative=False
    )

    # (1, 1, h, 21) -> the quantile row for horizon h
    row = forecast.quantiles[0, 0, int(h) - 1].to("cpu").numpy().astype(float)
    preds = np.interp(quantiles, model.quantile_levels, row)

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")


def qar_x(y, x, h=12, quantiles=QUANTILES, max_x_lags=MAX_X_LAGS):
    """Quantile ARX: quantiles of y_{t+h} from p lags of y and k lags of the FCI.

        Q(y_{t+h} | .) = a + sum_j b_j y_{t-j} + sum_l c_l x_{t-l}

    p is chosen as in quantile_ar (median-regression SIC on y alone), then k in
    1..max_x_lags by the same criterion with p fixed, so the FCI always enters.
    Estimated on the sample where the FCI is observed.
    """

    import statsmodels.api as sm

    from models.qar import select_lags

    y, x = _aligned(y, x)
    p = select_lags(y, h)

    y_lags = pd.concat([y.shift(j).rename(f"y_lag{j}") for j in range(p)], axis=1)
    x_lags = pd.concat([x.shift(j).rename(f"x_lag{j}") for j in range(max_x_lags)], axis=1)
    target = y.shift(-h).rename("target")

    # select k on one common sample (the one the largest k leaves)
    data = y_lags.join(x_lags).join(target).dropna()
    n = len(data)
    best_sic, best_k = np.inf, 1
    for k in range(1, max_x_lags + 1):
        X = sm.add_constant(data[list(y_lags.columns) + list(x_lags.columns[:k])], has_constant="add")
        resid = sm.QuantReg(data["target"], X).fit(q=0.5, max_iter=10000).resid
        sic = np.log(np.mean(np.abs(resid))) + (p + k + 1) * np.log(n) / (2 * n)
        if sic < best_sic:
            best_sic, best_k = sic, k

    X = sm.add_constant(y_lags.join(x_lags.iloc[:, :best_k]), has_constant="add")
    data = X.join(target).dropna()
    x_new = X.dropna().iloc[[-1]]

    preds = [
        sm.QuantReg(data["target"], data.drop(columns="target")).fit(q=tau, max_iter=5000)
           .predict(x_new).iloc[0]
        for tau in quantiles
    ]

    return pd.Series(np.sort(preds), index=quantiles, name="forecast")
