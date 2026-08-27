"""Proper scoring rules for quantile-represented predictive distributions.

All functions take `pred` with shape (..., n_quantiles) aligned to a strictly
increasing `quantiles` grid, and `y` broadcastable to `pred[..., 0]`.
"""

import numpy as np
from scipy import stats

# np.trapz was renamed np.trapezoid in numpy 2.0; keep both versions working.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def pinball_loss(y, pred, quantiles, average=True):
    """Quantile (pinball) loss.

        QL_tau(y, q) = (tau - 1{y < q}) * (y - q)

    With `average=False` the per-quantile losses are returned, which is what
    you want to see which part of the distribution a model is losing on.
    """

    y = np.asarray(y, dtype=float)[..., None]
    pred = np.asarray(pred, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    error = y - pred
    loss = np.where(error >= 0, quantiles * error, (quantiles - 1.0) * error)

    return loss.mean(axis=-1) if average else loss


def crps_from_quantiles(y, pred, quantiles):
    """CRPS approximated from a quantile grid.

    CRPS = 2 * integral_0^1 QL_tau dtau, so the average pinball loss over a
    grid, doubled, approximates it. The approximation improves with a denser
    grid and understates CRPS slightly because the tails beyond the outermost
    levels are not represented - a bias shared by every model scored on the
    same grid, so comparisons stay fair as long as the grid is held fixed.
    """

    return 2.0 * pinball_loss(y, pred, quantiles, average=True)


def predictive_mean(pred, quantiles):
    """The mean of a distribution represented by its quantiles.

        E[Y] = integral_0^1 Q(tau) dtau

    approximated by the trapezoid rule over the grid, plus rectangular tail
    pieces that hold Q flat beyond the outermost levels. With a 1%-99% grid the
    unrepresented tails carry 2% of the mass, so a heavy-tailed predictive
    distribution will have its mean pulled slightly toward the centre.

    Needed because absolute error and squared error elicit *different*
    functionals: the median minimises expected absolute error, the mean
    minimises expected squared error (Gneiting 2011, "Making and evaluating
    point forecasts"). Scoring a median forecast with squared error compares
    two things that are not the same target.
    """

    pred = np.asarray(pred, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    interior = _trapezoid(pred, quantiles, axis=-1)

    lower_tail = pred[..., 0] * quantiles[0]
    upper_tail = pred[..., -1] * (1.0 - quantiles[-1])

    return interior + lower_tail + upper_tail


def crps_skill_score(crps_model, crps_reference):
    """CRPS skill score against a reference forecast.

        CRPSS = 1 - CRPS_model / CRPS_reference

    Positive means the model beats the reference, 0 means no skill over it,
    negative means worse. Pair it with the climatological baseline
    (HistoricalQuantiles) for the conventional reading: skill over knowing
    nothing but the historical distribution of the series.

    Both arguments should already be averaged over whatever set of forecasts
    the comparison is about - a ratio of means, not a mean of ratios, which
    would be dominated by the origins where the reference happened to do well.
    """

    crps_reference = np.asarray(crps_reference, dtype=float)

    if np.any(crps_reference <= 0):
        raise ValueError("reference CRPS must be strictly positive")

    return 1.0 - np.asarray(crps_model, dtype=float) / crps_reference


def coverage(y, pred, quantiles, lower=0.05, upper=0.95):
    """Empirical frequency of y falling inside a nominal central interval."""

    quantiles = np.asarray(quantiles, dtype=float)
    pred = np.asarray(pred, dtype=float)

    i_lo = int(np.argmin(np.abs(quantiles - lower)))
    i_hi = int(np.argmin(np.abs(quantiles - upper)))

    if not np.isclose(quantiles[i_lo], lower) or not np.isclose(quantiles[i_hi], upper):
        raise ValueError(
            f"levels {lower} and {upper} are not both on the quantile grid"
        )

    y = np.asarray(y, dtype=float)
    inside = (y >= pred[..., i_lo]) & (y <= pred[..., i_hi])

    return inside.astype(float)


def tail_pinball(y, pred, quantiles, tau):
    """Pinball loss at one quantile level - the growth-at-risk loss function.

    CRPS scores the whole predictive distribution, which is the right summary
    when the question is "is this a good density forecast". It is not the right
    summary when the question is "is this a good estimate of the 5% worst
    case": a model can win on CRPS through the middle of the distribution while
    losing in the tail that the exercise is about. Brownlees & Souza (2021)
    backtest growth-at-risk on exactly this loss at tau = 0.05.

    `tau` must be on the quantile grid - interpolating a quantile and then
    scoring it with the pinball loss would score an object the model never
    produced.
    """

    quantiles = np.asarray(quantiles, dtype=float)
    index = int(np.argmin(np.abs(quantiles - tau)))

    if not np.isclose(quantiles[index], tau):
        raise ValueError(f"level {tau} is not on the quantile grid")

    y = np.asarray(y, dtype=float)
    q = np.asarray(pred, dtype=float)[..., index]

    error = y - q

    return np.where(error >= 0, tau * error, (tau - 1.0) * error)


def newey_west_variance(x, lags):
    """Long-run variance of the mean of x, Bartlett-weighted (Newey-West 1987).

    Returns Var(mean(x)), not the long-run variance of x itself, so the square
    root is directly a standard error.
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n < 2:
        return np.nan

    centred = x - x.mean()
    gamma0 = centred @ centred / n

    total = gamma0

    for lag in range(1, int(lags) + 1):

        if lag >= n:
            break

        gamma = centred[lag:] @ centred[:-lag] / n
        total += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma

    return total / n


def diebold_mariano(loss_a, loss_b, horizon=1):
    """Diebold-Mariano test on a paired loss differential (Diebold & Mariano 1995).

    d_t = loss_a - loss_b, H0: E[d] = 0. Negative statistic favours model a.

    The standard error is Newey-West with h - 1 lags, which is the whole point
    of using this rather than a t-test or a sign test: h-step-ahead forecast
    errors made at consecutive origins overlap, so d_t is autocorrelated by
    construction up to order h - 1 even when both models are perfectly
    specified. On a twelve-month-ahead target scored at monthly origins the
    naive standard error is understated by roughly the square root of the
    number of overlapping periods.

    Harvey, Leybourne & Newbold's (1997) small-sample correction is applied,
    and the statistic is referred to a t distribution with n - 1 degrees of
    freedom rather than the normal - with a few hundred overlapping
    observations the difference is not cosmetic.

    Returns (statistic, p_value, n), or NaN for the first two when the
    differential is degenerate - identical or constant losses have no sampling
    variance, and a 0/0 reported as a number would be read as evidence.

    Nested models are a known caveat: the DM
    test is not valid for a pair where one model nests the other under the
    null (Clark & West 2007), which applies to the covariate models here
    against their univariate counterparts. Read those as descriptive.
    """

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]

    n = len(d)

    if n < 3:
        return np.nan, np.nan, n

    variance = newey_west_variance(d, lags=max(int(horizon) - 1, 0))

    if not np.isfinite(variance) or variance <= 0:
        return np.nan, np.nan, n

    statistic = d.mean() / np.sqrt(variance)

    # Harvey-Leybourne-Newbold small-sample correction. It can go negative for
    # a horizon long relative to the sample, in which case the corrected
    # statistic is not defined and no number is reported.
    h = int(horizon)
    correction = (n + 1 - 2 * h + h * (h - 1) / n) / n

    if correction <= 0:
        return np.nan, np.nan, n

    statistic *= np.sqrt(correction)

    p_value = 2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1))

    return float(statistic), float(p_value), n


def coverage_tests(hits, level):
    """Kupiec and Christoffersen backtests of a quantile-exceedance sequence.

    `hits` is the 0/1 sequence 1{y_t < q_tau,t} and `level` is the nominal tau.
    This is the VaR backtesting apparatus, which applies unchanged to
    growth-at-risk: the lower predictive quantile *is* a value-at-risk number,
    and the same two questions are asked of it.

        LR_uc  Kupiec (1995) - unconditional coverage. Are exceedances as
               frequent as tau says?
        LR_ind Christoffersen (1998) - independence. Do exceedances cluster,
               i.e. does an exceedance today predict one tomorrow? A model can
               pass Kupiec and fail this badly, which is the interesting
               failure: it means the model has the average risk right and the
               timing wrong.
        LR_cc  the two combined, LR_uc + LR_ind, chi2(2).

    Both are asymptotic chi-square tests that assume *non-overlapping*
    forecasts. Feeding them an h-step target scored at every origin makes the
    independence test reject mechanically, so callers must thin to
    non-overlapping origins first. Returns NaN rather than a number where a
    transition count is empty (no exceedances at all, say), because a
    degenerate likelihood ratio is not a passing test result.
    """

    hits = np.asarray(hits, dtype=float)
    hits = hits[np.isfinite(hits)]

    n = len(hits)
    n1 = float(hits.sum())
    n0 = n - n1

    out = {
        "n": n,
        "expected_rate": float(level),
        "observed_rate": n1 / n if n else np.nan,
        "lr_uc": np.nan, "p_uc": np.nan,
        "lr_ind": np.nan, "p_ind": np.nan,
        "lr_cc": np.nan, "p_cc": np.nan
    }

    if n == 0 or n1 == 0 or n0 == 0:
        return out

    pi = n1 / n

    log_l0 = n0 * np.log(1 - level) + n1 * np.log(level)
    log_l1 = n0 * np.log(1 - pi) + n1 * np.log(pi)

    lr_uc = -2.0 * (log_l0 - log_l1)

    out["lr_uc"] = float(lr_uc)
    out["p_uc"] = float(1.0 - stats.chi2.cdf(lr_uc, df=1))

    # Transition counts for the first-order Markov alternative.
    previous, current = hits[:-1], hits[1:]

    n00 = float(((previous == 0) & (current == 0)).sum())
    n01 = float(((previous == 0) & (current == 1)).sum())
    n10 = float(((previous == 1) & (current == 0)).sum())
    n11 = float(((previous == 1) & (current == 1)).sum())

    if (n00 + n01) == 0 or (n10 + n11) == 0 or n11 == 0:
        return out

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi_all = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi01 in (0.0, 1.0) or pi11 in (0.0, 1.0):
        return out

    log_l_markov = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01)
        + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    log_l_iid = (
        (n00 + n10) * np.log(1 - pi_all) + (n01 + n11) * np.log(pi_all)
    )

    lr_ind = -2.0 * (log_l_iid - log_l_markov)
    lr_cc = lr_uc + lr_ind

    out["lr_ind"] = float(lr_ind)
    out["p_ind"] = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    out["lr_cc"] = float(lr_cc)
    out["p_cc"] = float(1.0 - stats.chi2.cdf(lr_cc, df=2))

    return out


def exceedances(y, pred, quantiles, tau):
    """The 0/1 sequence 1{y < q_tau}, for the coverage tests above."""

    quantiles = np.asarray(quantiles, dtype=float)
    index = int(np.argmin(np.abs(quantiles - tau)))

    if not np.isclose(quantiles[index], tau):
        raise ValueError(f"level {tau} is not on the quantile grid")

    y = np.asarray(y, dtype=float)

    return (y < np.asarray(pred, dtype=float)[..., index]).astype(float)


def pit_from_quantiles(y, pred, quantiles):
    """Probability integral transform, by interpolating the quantile function.

    Returns the level tau at which the predictive quantile equals y. Under a
    correctly calibrated forecast these are uniform on (0, 1); histogram them
    to read off bias (shifted), overconfidence (U-shaped) or underconfidence
    (hump-shaped). Realisations beyond the outermost grid levels are clipped
    there, so a mass of exact 0.05 / 0.95 values is itself a tail-miss signal.
    """

    y = np.atleast_1d(np.asarray(y, dtype=float))
    pred = np.asarray(pred, dtype=float).reshape(-1, len(quantiles))
    quantiles = np.asarray(quantiles, dtype=float)

    out = np.empty(len(y))

    for i in range(len(y)):
        out[i] = np.interp(y[i], pred[i], quantiles)

    return out
