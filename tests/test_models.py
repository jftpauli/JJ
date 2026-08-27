"""Correctness checks against cases with a known answer."""

import numpy as np
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Chronos2, HistoricalQuantiles, QuantileAR, DEFAULT_QUANTILES
from models.base import direct_design, prediction_row, seasonal_dummies

rng = np.random.default_rng(0)
ok = True


def check(name, condition, detail=""):
    global ok
    print(("  PASS  " if condition else "  FAIL  ") + name + ("  " + detail if detail else ""))
    ok = ok and bool(condition)


print("\n[1] direct_design / prediction_row alignment")
y = np.arange(100.0)
X, z = direct_design(y, p=3, h=2)
# origins t = 2..97, features [y_t, y_t-1, y_t-2], target y_{t+2}
check("first row features", np.allclose(X[0], [1, 2, 1, 0]), str(X[0]))
check("first row target", z[0] == 4.0, str(z[0]))
check("last row target", z[-1] == 99.0, str(z[-1]))
# origins t run p-1=2 .. n-h-1=97 inclusive -> 96 rows
check("n rows", len(z) == 96, str(len(z)))
x0 = prediction_row(y, p=3, h=2)
check("prediction row = latest lags", np.allclose(x0, [1, 99, 98, 97]), str(x0))

print("\n[2] seasonal dummies")
d = seasonal_dummies(np.array([0, 1, 11, 12]), 12)
check("season 0 is reference (all zero)", d[0].sum() == 0)
check("season 1 sets first dummy", d[1][0] == 1 and d[1].sum() == 1)
check("season 11 sets last dummy", d[2][-1] == 1 and d[2].sum() == 1)
check("position 12 wraps to season 0", d[3].sum() == 0)

print("\n[3] HistoricalQuantiles reproduces the empirical distribution")
draw = rng.normal(loc=5.0, scale=2.0, size=4000)
taus = [0.1, 0.5, 0.9]
hq = HistoricalQuantiles(quantiles=taus).fit(draw)
hq_out = hq.predict_quantiles(6)
check("matches np.quantile of the history",
      np.allclose(hq_out[0], np.quantile(draw, taus)))
check("identical at every horizon",
      np.allclose(hq_out, hq_out[0]), "horizon-invariant by construction")
check("recovers the true N(5,2) quantiles",
      np.max(np.abs(hq_out[0] - stats.norm.ppf(taus, 5.0, 2.0))) < 0.15,
      f"max err {np.max(np.abs(hq_out[0] - stats.norm.ppf(taus, 5.0, 2.0))):.3f}")

print("\n[4] HistoricalQuantiles window and seasonal options")
# Level shift: the last 100 points are drawn around 50, the first 400 around 0.
shifted = np.concatenate([rng.normal(0, 1, 400), rng.normal(50, 1, 100)])
full = HistoricalQuantiles(quantiles=[0.5]).fit(shifted).predict_quantiles(1)
windowed = HistoricalQuantiles(window=100, quantiles=[0.5]).fit(shifted).predict_quantiles(1)
check("full history straddles the shift", abs(full[0, 0] - 0.0) < 3.0, f"{full[0,0]:.2f}")
check("100-obs window tracks the new level", abs(windowed[0, 0] - 50.0) < 1.0,
      f"{windowed[0,0]:.2f}")
# Seasonal climatology: value == month number, so each season is degenerate.
months = np.tile(np.arange(1, 13), 30).astype(float)  # starts Jan, ends Dec
seas = HistoricalQuantiles(seasonal_period=12, quantiles=[0.5]).fit(months)
seas_out = seas.predict_quantiles(12)
check("h=1 picks the January climatology", np.isclose(seas_out[0, 0], 1.0),
      str(seas_out[0, 0]))
check("h=6 picks the June climatology", np.isclose(seas_out[5, 0], 6.0),
      str(seas_out[5, 0]))
check("h=12 picks the December climatology", np.isclose(seas_out[11, 0], 12.0),
      str(seas_out[11, 0]))

print("\n[5] HistoricalQuantiles recomputes from a later origin")
# Use an upper quantile: the added 100 high values are 20% of the extended
# sample, so they move tau=0.9 decisively while barely shifting the median.
hq2 = HistoricalQuantiles(quantiles=[0.9]).fit(shifted[:400])
before = hq2.predict_quantiles(1)[0, 0]
after = hq2.predict_quantiles(1, history=shifted)[0, 0]
check("history updates the distribution", before < 3.0 and after > 45.0,
      f"{before:.2f} -> {after:.2f}")

print("\n[6] CRPS skill score")
from models import crps_skill_score
check("zero skill against itself", np.isclose(crps_skill_score(1.0, 1.0), 0.0))
check("halving CRPS gives 0.5 skill", np.isclose(crps_skill_score(0.5, 1.0), 0.5))
check("worse than reference is negative", crps_skill_score(2.0, 1.0) < 0)

print("\n[7] QAR recovers known conditional quantiles")
# y_t = 0.5 y_{t-1} + e,  e ~ N(0,1): true tau-quantile = 0.5*y_T + z_tau
n = 6000
e = rng.normal(size=n)
s = np.zeros(n)
for t in range(1, n):
    s[t] = 0.5 * s[t - 1] + e[t]
qar = QuantileAR(p=1, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]).fit(s)
qp = qar.predict_quantiles(1)[0]
truth = 0.5 * s[-1] + stats.norm.ppf([0.1, 0.25, 0.5, 0.75, 0.9])
check("quantiles within 0.12 of truth", np.max(np.abs(qp - truth)) < 0.12,
      f"max err {np.max(np.abs(qp - truth)):.3f}")
check("no crossing", np.all(np.diff(qp) > 0))
# LP solution must agree with a converged IRLS fit (statsmodels optional)
try:
    from statsmodels.regression.quantile_regression import QuantReg
except ImportError:
    print("  SKIP  LP vs IRLS cross-check (statsmodels not installed)")
else:
    from models.base import direct_design
    from models.qar import quantile_regression
    Xd, zd = direct_design(s, 1, 1)
    lp = np.array([quantile_regression(Xd, zd, t) for t in [0.1, 0.5, 0.9]])
    irls = np.array([QuantReg(zd, Xd).fit(q=t).params for t in [0.1, 0.5, 0.9]])
    check("LP agrees with converged IRLS", np.abs(lp - irls).max() < 1e-4,
          f"max diff {np.abs(lp-irls).max():.2e}")

print("\n[8] monotonicity + shape contract for both models")
data = np.cumsum(rng.normal(size=300)) + 100
for name, mdl in [
    ("HistoricalQuantiles", HistoricalQuantiles()),
    ("QuantileAR", QuantileAR(p=3)),
]:
    out = mdl.fit(data).predict_quantiles(12)
    check(f"{name} shape (12, {len(DEFAULT_QUANTILES)})",
          out.shape == (12, len(DEFAULT_QUANTILES)), str(out.shape))
    check(f"{name} finite", np.isfinite(out).all())
    check(f"{name} non-decreasing across quantiles",
          np.all(np.diff(out, axis=1) >= -1e-9))

print("\n[9] NaN handling")
gappy = data.copy()
gappy[150] = np.nan
out = QuantileAR(p=2).fit(gappy).predict_quantiles(4)
check("QAR tolerates an interior NaN (row dropped)", np.isfinite(out).all())
out = HistoricalQuantiles().fit(gappy).predict_quantiles(4)
check("HistoricalQuantiles drops NaN from the sample", np.isfinite(out).all())
tail_nan = data.copy()
tail_nan[-1] = np.nan
try:
    QuantileAR(p=2).fit(tail_nan).predict_quantiles(1)
    check("QAR raises on NaN at the origin", False)
except ValueError as err:
    check("QAR raises informatively on NaN at the origin", "NaN" in str(err))
# The unconditional baseline conditions on nothing, so a NaN at the origin is
# not a problem for it - it must still produce a forecast.
out = HistoricalQuantiles().fit(tail_nan).predict_quantiles(1)
check("HistoricalQuantiles unaffected by NaN at the origin", np.isfinite(out).all())

print("\n[10] Chronos2 obeys the same contract")
# Skipped rather than failed when the weights are not available: the rest of
# the suite must still run on a machine that has never downloaded them, and a
# missing checkpoint is an environment fact, not a defect in this code.
try:
    chronos = Chronos2(quantiles=DEFAULT_QUANTILES).fit(data)
except Exception as err:  # noqa: BLE001 - any load failure means "not available"
    print(f"  SKIP  Chronos-2 weights unavailable ({type(err).__name__}: {err})")
else:
    out = chronos.predict_quantiles(12)
    check(f"shape (12, {len(DEFAULT_QUANTILES)})",
          out.shape == (12, len(DEFAULT_QUANTILES)), str(out.shape))
    check("finite", np.isfinite(out).all())
    check("non-decreasing across quantiles",
          np.all(np.diff(out, axis=1) >= -1e-9))

    # No sampling anywhere in the forward pass, so two calls on the same
    # history must agree exactly. If this ever fails, every score in the
    # backtest becomes irreproducible.
    check("deterministic across calls",
          np.array_equal(out, chronos.predict_quantiles(12)))

    # Fitted on a short prefix, predicted from a longer history: the forecast
    # must move with the history, since there is no fitted state to fall back
    # on. This is the property the rolling-origin backtest relies on.
    short = Chronos2(quantiles=DEFAULT_QUANTILES).fit(data[:150])
    check("recomputes from a later origin",
          not np.allclose(
              short.predict_quantiles(1),
              short.predict_quantiles(1, history=data)
          ))
    check("later origin matches a fresh fit on the same history",
          np.allclose(
              short.predict_quantiles(1, history=data),
              chronos.predict_quantiles(1)
          ))

    # Unlike QuantileAR, the foundation model handles a NaN at the origin
    # natively - it is a documented capability, and run_benchmarks.py relies
    # on it not raising.
    out = chronos.predict_quantiles(1, history=tail_nan)
    check("tolerates NaN at the forecast origin", np.isfinite(out).all())

    # Too short a history is refused rather than answered from four points.
    try:
        Chronos2(min_observations=20).fit(data[:5])
        check("raises on too short a history", False)
    except ValueError:
        check("raises on too short a history", True)

print("\n[11] covariates enter at the origin, never from the future")
# y_{t+1} = 2 x_t + e. A model conditioning on x must recover the coefficient;
# the univariate model cannot see it at all.
n = 4000
x = rng.normal(size=n)
noise = rng.normal(size=n)
yx = np.empty(n)
yx[0] = noise[0]
yx[1:] = 2.0 * x[:-1] + noise[1:]

Xd, zd = direct_design(yx, p=1, h=1, exog=x[:, None])
# Row t of the design must carry x_t, not x_{t+1}: the origin's value.
check("design takes the covariate at the origin",
      np.allclose(Xd[:, -1], x[: len(zd)]), str(Xd[0]))
check("design target is y_{t+1}", np.allclose(zd, yx[1: len(zd) + 1]))

x0 = prediction_row(yx, p=1, h=1, exog=x[:, None])
check("prediction row takes the last covariate", np.isclose(x0[-1], x[-1]))

taus = [0.1, 0.5, 0.9]
cond = QuantileAR(p=1, quantiles=taus, uses_exog=True).fit(yx, exog=x[:, None])
cond_out = cond.predict_quantiles(1, history=yx, exog_history=x[:, None])[0]
truth = 2.0 * x[-1] + stats.norm.ppf(taus)
check("conditional QAR recovers 2*x_T + z_tau",
      np.max(np.abs(cond_out - truth)) < 0.15,
      f"max err {np.max(np.abs(cond_out - truth)):.3f}")

uncond_out = QuantileAR(p=1, quantiles=taus).fit(yx).predict_quantiles(1)[0]
check("univariate QAR misses it (x is unobserved to it)",
      np.max(np.abs(uncond_out - truth)) > 0.3,
      f"max err {np.max(np.abs(uncond_out - truth)):.3f}")

# A model that declares covariates must not silently estimate a different one.
try:
    QuantileAR(p=1, uses_exog=True).fit(yx)
    check("QAR raises when declared covariates are not supplied", False)
except ValueError:
    check("QAR raises when declared covariates are not supplied", True)

try:
    QuantileAR(p=1, uses_exog=True).fit(yx, exog=x[:50, None])
    check("QAR raises on a length mismatch", False)
except ValueError:
    check("QAR raises on a length mismatch", True)

try:
    QuantileAR(p=1).fit(yx, exog=x[:, None])
    check("QAR raises on covariates it was not built for", False)
except ValueError:
    check("QAR raises on covariates it was not built for", True)

# A NaN in the covariate at the origin is skipped, not imputed.
gappy_x = x.copy()
gappy_x[-1] = np.nan
try:
    cond.predict_quantiles(1, history=yx, exog_history=gappy_x[:, None])
    check("QAR raises on NaN covariate at the origin", False)
except ValueError:
    check("QAR raises on NaN covariate at the origin", True)

print("\n[12] tail scoring, HAC inference and the coverage backtests")
from models import (
    coverage_tests,
    diebold_mariano,
    exceedances,
    newey_west_variance,
    pinball_loss,
    tail_pinball,
)

grid = np.array([0.05, 0.5, 0.95])
preds = np.array([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]])
actuals = np.array([-3.0, 2.0])

# tail_pinball at one level must equal that column of the full pinball loss.
per_level = pinball_loss(actuals, preds, grid, average=False)
check("tail pinball matches the 5% column of the pinball loss",
      np.allclose(tail_pinball(actuals, preds, grid, 0.05), per_level[:, 0]))
try:
    tail_pinball(actuals, preds, grid, 0.07)
    check("tail pinball raises off-grid", False)
except ValueError:
    check("tail pinball raises off-grid", True)

check("exceedances flag realisations below the quantile",
      np.array_equal(exceedances(actuals, preds, grid, 0.05), [1.0, 0.0]))

# Newey-West must widen the variance on positively autocorrelated data and
# agree with the iid formula when there is no autocorrelation.
iid = rng.normal(size=3000)
check("NW with 0 lags is the iid variance of the mean",
      np.isclose(newey_west_variance(iid, 0), iid.var() / len(iid), rtol=1e-8))

ar = np.empty(3000)
ar[0] = rng.normal()
for t in range(1, 3000):
    ar[t] = 0.8 * ar[t - 1] + rng.normal()
check("NW inflates the variance on autocorrelated data",
      newey_west_variance(ar, 11) > 3.0 * newey_west_variance(ar, 0),
      f"{newey_west_variance(ar, 11):.4f} vs {newey_west_variance(ar, 0):.4f}")

# DM: identical losses cannot favour either model; a uniformly better model
# must be favoured; and the HAC version must be more conservative than the
# naive one on overlapping forecasts.
loss_a = rng.gamma(2.0, 1.0, size=500)

# A degenerate differential has no sampling variance, so there is no test to
# run. NaN is the correct answer; a 0/0 that came back as a number would be
# read as evidence.
statistic, p_value, _ = diebold_mariano(loss_a, loss_a, horizon=1)
check("DM returns no statistic for identical losses",
      np.isnan(statistic) and np.isnan(p_value))
check("DM returns no statistic for a constant differential",
      np.isnan(diebold_mariano(loss_a - 1.0, loss_a, horizon=1)[0]))

loss_b = loss_a + rng.normal(1.0, 1.0, size=500)
statistic, p_value, _ = diebold_mariano(loss_a, loss_b, horizon=1)
check("DM favours the better model", statistic < 0 and p_value < 0.01,
      f"t = {statistic:.2f}, p = {p_value:.1e}")

overlapping = 0.3 + ar * 0.5
naive = diebold_mariano(overlapping, np.zeros_like(overlapping), horizon=1)[0]
hac = diebold_mariano(overlapping, np.zeros_like(overlapping), horizon=12)[0]
check("HAC is more conservative than the naive statistic on overlap",
      abs(hac) < abs(naive), f"|{hac:.2f}| < |{naive:.2f}|")

# Kupiec / Christoffersen on a hit sequence with a known rate.
calibrated = (rng.random(2000) < 0.05).astype(float)
tests = coverage_tests(calibrated, 0.05)
check("Kupiec does not reject a correctly calibrated 5% quantile",
      tests["p_uc"] > 0.05, f"rate {tests['observed_rate']:.3f}, "
      f"p = {tests['p_uc']:.3f}")
check("Christoffersen does not reject independent hits",
      tests["p_ind"] > 0.05, f"p = {tests['p_ind']:.3f}")

overconfident = (rng.random(2000) < 0.20).astype(float)
check("Kupiec rejects a 5% quantile breached 20% of the time",
      coverage_tests(overconfident, 0.05)["p_uc"] < 1e-6)

# Clustered hits at the right average rate: Kupiec passes, independence fails.
clustered = np.zeros(2000)
for start in range(0, 2000, 200):
    clustered[start: start + 10] = 1.0
clustered_tests = coverage_tests(clustered, 0.05)
check("clustered hits pass Kupiec (the rate is right)",
      clustered_tests["p_uc"] > 0.05, f"p = {clustered_tests['p_uc']:.3f}")
check("clustered hits fail Christoffersen (the timing is not)",
      clustered_tests["p_ind"] < 1e-6, f"p = {clustered_tests['p_ind']:.2e}")

check("degenerate hit sequences return NaN rather than a number",
      np.isnan(coverage_tests(np.zeros(100), 0.05)["lr_uc"]))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
