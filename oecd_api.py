"""Build CPI and real GDP panels from the OECD SDMX API.

Every series key below is fully specified - no wildcard positions. A wildcard
can return several activities, price bases and adjustment modes at once, and
collapsing that with pivot_table(aggfunc="first") silently picks an arbitrary
one per country (and, worse, a different one in different periods). The fetch
here refuses to guess: if a key does not resolve to exactly one observation per
country-period, it raises.

Panels written to data/:

  OECD_cpi_index_monthly_panel.csv     CPI, all items, index, not seas. adj.
  OECD_cpi_yoy_monthly_panel.csv       CPI, all items, % change on a year earlier
  OECD_gdp_quarterly_panel.csv         Real GDP, chain-linked volume, seas. adj.
  OECD_gdp_growth_quarterly_panel.csv  Real GDP, quarter-on-quarter % (dlog)

The two variables run at *different frequencies* - CPI monthly, GDP quarterly -
so a forecast horizon h means months for one and quarters for the other.
Anything consuming these panels has to carry that distinction; run_benchmarks.py
does so through a per-target frequency setting.

Each variable is written twice: once as the level and once as the growth rate
the forecasting work actually consumes. The growth panels are ready to model as
they stand, so nothing downstream has to transform anything; the level panels
stay because a level cannot be recovered from a growth rate, and they are what
any later change of transformation has to start from.
"""

import io
import os
from datetime import date

import numpy as np
import pandas as pd
import requests


# ======================================================
# USER SETTINGS
# ======================================================

COUNTRIES = [
    "USA",
    "DEU",
    "GBR",
    "FRA"
]

START_DATE = "1990-01"
END_DATE = "2025-12"

OUTPUT_FOLDER = "data"

OUTPUT_FILE_CPI_INDEX = "OECD_cpi_index_monthly_panel.csv"
OUTPUT_FILE_CPI_YOY = "OECD_cpi_yoy_monthly_panel.csv"
OUTPUT_FILE_GDP = "OECD_gdp_quarterly_panel.csv"
OUTPUT_FILE_GDP_GROWTH = "OECD_gdp_growth_quarterly_panel.csv"

REQUEST_TIMEOUT = 180


# ======================================================
# SERIES DEFINITIONS
# ======================================================
#
# CPI key : REF_AREA.FREQ.METHODOLOGY.MEASURE.UNIT_MEASURE.EXPENDITURE.
#           ADJUSTMENT.TRANSFORMATION
# GDP key : FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.
#           INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.
#           TRANSFORMATION.TABLE_IDENTIFIER   (13 positions)
#
# Choices and why:
#   * National CPI methodology (N), not HICP - HICP does not cover the US, so
#     the national measure is the only definition consistent across all four.
#   * CPI not seasonally adjusted - a seasonally adjusted index is not
#     published for the UK, so NSA is the only consistent choice. Seasonally
#     adjust downstream if the application needs it.
#   * All items (_T) - the headline basket.
#   * GDP as B1GQ at chain-linked volumes (PRICE_BASE = L), i.e. *real* GDP.
#     Nominal GDP (V) would fold inflation into the output target and overlap
#     with the CPI series.
#   * Seasonally adjusted (ADJUSTMENT = Y) - the published form, and the only
#     one where a quarter-on-quarter growth rate is meaningful.
#   * National currency (XDC). The base year differs across countries, which
#     does not matter because the modelling target is a growth rate.

CPI_FLOW = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,"
GDP_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,"

CPI_INDEX_KEY = "{countries}.M.N.CPI.IX._T.N._Z"
CPI_YOY_KEY = "{countries}.M.N.CPI.PA._T.N.GY"
GDP_KEY = "Q.Y.{countries}...B1GQ....XDC.L.."

SERIES = [
    {
        "name": "CPI index (all items, NSA)",
        "flow": CPI_FLOW,
        "key": CPI_INDEX_KEY,
        "filter": {},
        "file": OUTPUT_FILE_CPI_INDEX,
        "freq": "M"
    },
    {
        "name": "CPI year-on-year (%)",
        "flow": CPI_FLOW,
        "key": CPI_YOY_KEY,
        "filter": {},
        "file": OUTPUT_FILE_CPI_YOY,
        "freq": "M"
    },
    {
        "name": "Real GDP, quarterly (B1GQ, chain-linked volume, seas. adj.)",
        "flow": GDP_FLOW,
        "key": GDP_KEY,
        "filter": {},
        "file": OUTPUT_FILE_GDP,
        "freq": "Q"
    }
]


# ======================================================
# FETCH
# ======================================================

def period_bounds(freq):
    """SDMX period strings for the configured span, in the flow's frequency."""

    if freq == "M":
        return START_DATE, END_DATE

    if freq == "Q":
        start = pd.Period(START_DATE, freq="M").asfreq("Q", how="start")
        end = pd.Period(END_DATE, freq="M").asfreq("Q", how="end")
        return f"{start.year}-Q{start.quarter}", f"{end.year}-Q{end.quarter}"

    raise ValueError(f"unsupported frequency: {freq}")


def parse_periods(index, freq):
    """Turn SDMX period labels into timestamps at the start of each period.

    Quarterly labels look like '1990-Q1', which pd.to_datetime does not accept;
    they have to go through PeriodIndex.
    """

    if freq == "Q":
        return pd.PeriodIndex(index, freq="Q").to_timestamp(how="start")

    return pd.to_datetime(index)


def fetch(flow, key, selection, freq="M"):
    """Download one slice and pivot it to a country panel.

    Raises if the requested slice is not unique per country-period - an
    ambiguous key must never be resolved by silently taking the first row.
    """

    start, end = period_bounds(freq)

    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        f"{flow}/"
        f"{key.format(countries='+'.join(COUNTRIES))}?"
        f"startPeriod={start}"
        f"&endPeriod={end}"
        "&format=csvfilewithlabels"
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text),
        low_memory=False
    )

    for column, value in selection.items():
        df = df[df[column] == value]

    if df.empty:
        raise ValueError(f"No observations returned for key: {key}")

    duplicates = df.duplicated(
        subset=["REF_AREA", "TIME_PERIOD"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate country-month rows for key {key} - the "
            "slice is ambiguous. Add dimensions to the filter."
        )

    panel = df.pivot(
        index="TIME_PERIOD",
        columns="REF_AREA",
        values="OBS_VALUE"
    )

    panel.index = parse_periods(panel.index, freq)
    panel = panel.sort_index()

    # Stable column order, and a hard check that every country came back.
    missing = set(COUNTRIES) - set(panel.columns)

    if missing:
        raise ValueError(f"No data returned for: {sorted(missing)}")

    panel = panel[sorted(COUNTRIES)]
    panel.columns.name = None

    return panel


def report(panel, name, freq="M"):
    """Print the coverage facts worth knowing before the data is used."""

    # Gaps are counted inside the panel's own span. A derived series starts one
    # period late by construction, which is not a hole - the span line below
    # already makes the start date visible.
    expected = pd.date_range(
        start=panel.index.min(),
        end=panel.index.max(),
        freq={"M": "MS", "Q": "QS"}[freq]
    )

    gaps = expected.difference(panel.index)

    print(f"\n{name}")
    print(f"  span      : {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}"
          f"  ({len(panel)} periods)")
    print(f"  gaps      : {len(gaps)} missing months")
    print(f"  missing   : {panel.isna().sum().to_dict()}")
    print(f"  last obs  : {panel.ffill().iloc[-1].round(2).to_dict()}")

    if len(gaps):
        print(f"  gap dates : {[f'{d:%Y-%m}' for d in gaps[:12]]}")


# ======================================================
# MAIN
# ======================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

panels = {}

for spec in SERIES:

    print(f"\nFetching {spec['name']} ...")

    panel = fetch(
        spec["flow"],
        spec["key"],
        spec["filter"],
        spec["freq"]
    )

    report(panel, spec["name"], spec["freq"])

    output_path = os.path.join(
        OUTPUT_FOLDER,
        spec["file"]
    )

    panel.to_csv(output_path)
    panels[spec["file"]] = panel

    print("  saved     :", output_path)


# ======================================================
# DERIVED SERIES
# ======================================================
#
# Real GDP as a quarter-on-quarter growth rate, so downstream work never has to
# transform it. Derived from the level above rather than fetched separately,
# which guarantees the two are internally consistent and pins the definition
# (log difference, not a simple percentage change - log differences are
# additive across periods and symmetric in sign, which is what a forecasting
# target should be).
#
# The level panel is kept as well. A level cannot be recovered from a growth
# rate, so it stays the primitive, exactly as the CPI index does.

print("\nDeriving real GDP growth ...")

gdp_growth = 100.0 * np.log(panels[OUTPUT_FILE_GDP]).diff()

# The first quarter is structurally NaN (nothing to difference against).
gdp_growth = gdp_growth.iloc[1:]

report(
    gdp_growth,
    "Real GDP growth (100 * dlog, quarter on quarter %)",
    "Q"
)

growth_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE_GDP_GROWTH)
gdp_growth.to_csv(growth_path)

print("  saved     :", growth_path)


print(f"\nFinished. Vintage: {date.today():%Y-%m-%d} "
      "(OECD revises past observations - record this with any results).")
