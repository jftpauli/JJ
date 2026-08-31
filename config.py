"""Settings from config.toml. Paths are resolved against the repository root."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with open(ROOT / "config.toml", "rb") as f:
    _cfg = tomllib.load(f)

COUNTRIES = _cfg["COUNTRIES"]
LOOKBACK = _cfg["LOOKBACK"]
N_SAMPLES = _cfg["N_SAMPLES"]
FORECAST_HORIZONS = _cfg["FORECAST_HORIZONS"]
QUANTILES = tuple(_cfg["QUANTILES"])
FORECAST_START_DATE = _cfg["FORECAST_START_DATE"]
DATA_PATHS = {k: ROOT / v for k, v in _cfg["DATA_PATHS"].items()}
RESULTS_PATH = ROOT / _cfg["RESULTS_PATH"]
