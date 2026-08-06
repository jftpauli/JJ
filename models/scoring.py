"""Proper scoring rules for quantile-represented predictive distributions.

All functions take `pred` with shape (..., n_quantiles) aligned to a strictly
increasing `quantiles` grid, and `y` broadcastable to `pred[..., 0]`.
"""

import numpy as np

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
