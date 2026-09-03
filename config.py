"""Settings from config.toml. Paths are resolved against the repository root."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent

with open(ROOT / "config.toml", "rb") as f:
    _cfg = tomllib.load(f)

COUNTRIES = _cfg["COUNTRIES"]
INDICATORS = _cfg["INDICATORS"]

LOOKBACK = _cfg["LOOKBACK"]
N_SAMPLES = _cfg["N_SAMPLES"]
FORECAST_HORIZONS = _cfg["FORECAST_HORIZONS"]
QUANTILES = tuple(_cfg["QUANTILES"])
FORECAST_START_DATE = _cfg["FORECAST_START_DATE"]

DATA_PATHS = {k: ROOT / v for k, v in _cfg["DATA_PATHS"].items()}
RESULTS_PATH = ROOT / _cfg["RESULTS_PATH"]

BISTRO_ROOT = Path(_cfg["BISTRO_ROOT"])