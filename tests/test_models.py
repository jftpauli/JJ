"""Correctness checks against cases with a known answer."""

import numpy as np
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import HistoricalQuantiles, QuantileAR, DEFAULT_QUANTILES
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

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
