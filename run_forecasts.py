"""Rolling-origin quantile forecasts for the CPI panel -> results/quantile_forecasts.csv."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from models.qar import LAGS, QUANTILES

PANEL, OUTPUT, START = ("data/OECD_cpi_yoy_monthly_panel.csv",
                        "results/quantile_forecasts.csv", "2005-01-01")


def backtest(y, h, p):
    """Forecast y_{t+h} from every origin, refitting on what was known at each."""

    y = y.dropna()
    X = sm.add_constant(pd.concat([y.rename(j).shift(j) for j in range(p)], axis=1))
    frame = X.join(y.shift(-h).rename("target"))
    out = {}

    for i, origin in enumerate(y.index):

        if origin < pd.Timestamp(START) or not X.iloc[i].notna().all():
            continue

        train = frame.iloc[:i - h + 1].dropna()   # targets observed by the origin
        out[origin] = np.sort([
            sm.QuantReg(train["target"], train.drop(columns="target")).fit(q=tau)
              .predict(X.iloc[[i]]).iloc[0]
            for tau in QUANTILES
        ])

    forecasts = pd.DataFrame(out, index=[f"q{tau}" for tau in QUANTILES]).T
    return forecasts.assign(horizon=h, actual=y.shift(-h).reindex(forecasts.index))


panel = pd.read_csv(PANEL, index_col=0, parse_dates=True)

pd.concat(
    {c: pd.concat([backtest(panel[c], h, LAGS[h]) for h in LAGS]) for c in panel},
    names=["country", "origin"]
).to_csv(OUTPUT, float_format="%.6f")

print("saved", OUTPUT)
