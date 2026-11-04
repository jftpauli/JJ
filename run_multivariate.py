"""Multivariate pipeline: rolling-origin quantile forecasts of CPI and IPI with a
financial conditions index (FCI) as past-only covariate
-> results/multivariate/<model>_<indicator>_forecasts.csv

Models: chronos2, t0gamma, timestral (zero-shot, FCI as covariate) and qar
(quantile AR plus FCI lags) as the benchmark. See models/multivariate.py.

The output files have the same layout as results/<model>_<indicator>_forecasts.csv,
so the univariate runs are the natural "without FCI" comparison.

Usage:
    python run_multivariate.py                 # all models, each in its own venv
    python run_multivariate.py qar chronos2    # a subset
    python run_multivariate.py --worker <model>   # internal: one model, current interpreter
"""

import subprocess
import sys

import pandas as pd

from config import COUNTRIES, DATA_PATHS, FCI_COLUMN, INDICATORS, RESULTS_PATH

OUT_PATH = RESULTS_PATH / "multivariate"

MODEL_NAMES = ["qar", "chronos2", "t0gamma", "timestral"]

FORECASTERS = {
    "qar": "qar_x",
    "chronos2": "chronos2_x",
    "t0gamma": "t0gamma_x",
    "timestral": "timestral_x",
}


def load_fci(column=FCI_COLUMN):
    """FCI panel in the layout of the OECD panels: monthly index, one column per country."""

    fci = pd.read_csv(DATA_PATHS["fci"], usecols=["date", "iso_alpha3", column])
    fci["date"] = pd.to_datetime(fci["date"], format="%m/%d/%Y")
    return fci.pivot(index="date", columns="iso_alpha3", values=column)[COUNTRIES]


def worker(model_name):
    from forecast_worker import country_forecasts, print_progress
    import models.multivariate as mv

    f = getattr(mv, FORECASTERS[model_name])
    fci = load_fci()
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    for indicator in INDICATORS:
        panel = pd.read_csv(DATA_PATHS[indicator], index_col=0, parse_dates=True)[COUNTRIES]

        results = {}
        for i, c in enumerate(panel, start=1):
            print_progress(f"{model_name}+fci/{indicator}", i, len(panel.columns))
            x = fci[c]
            # backtest hands the forecaster y up to the origin; x is cut to the same dates
            forecast = lambda y, h, x=x: f(y, x.reindex(y.index), h=h)
            results[c] = country_forecasts(forecast, panel[c])

        results = {c: r for c, r in results.items() if not r.empty}
        path = OUT_PATH / f"{model_name}_{indicator}_forecasts.csv"
        pd.concat(results, names=["country", "origin"]).to_csv(path, float_format="%.6f")
        print("saved", path)


def main(model_names):
    from run_forecasts import VENVS, child_env

    status = {}
    for i, name in enumerate(model_names, start=1):
        python_exe = VENVS.get(name) or sys.executable
        print(f"[{i}/{len(model_names)}] running {name}+fci ({python_exe})", flush=True)
        code = subprocess.run(
            [str(python_exe), __file__, "--worker", name], cwd=RESULTS_PATH.parent, env=child_env(python_exe)
        ).returncode
        status[name] = code
        print(f"[{i}/{len(model_names)}] {name}: {'ok' if code == 0 else f'FAILED (exit {code})'}", flush=True)

    print("\nsummary:")
    for name, code in status.items():
        print(f"  {name}: {'ok' if code == 0 else f'FAILED (exit {code})'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--worker"]:
        worker(args[1])
    else:
        unknown = set(args) - set(FORECASTERS)
        if unknown:
            sys.exit(f"unknown model(s): {sorted(unknown)}; choose from {MODEL_NAMES}")
        main(args or MODEL_NAMES)
