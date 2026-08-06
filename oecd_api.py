"""Build monthly CPI and industrial-production panels from the OECD SDMX API.

Every series key below is fully specified - no wildcard positions. A wildcard
returns several economic activities and adjustment modes at once, and collapsing
that with pivot_table(aggfunc="first") silently picks an arbitrary one per
country (and, worse, a different one in different months). The fetch here
refuses to guess: if a key does not resolve to exactly one observation per
country-month, it raises.

Panels written to data/:

  OECD_cpi_index_monthly_panel.csv  CPI, all items, index, not seasonally adj.
  OECD_cpi_yoy_monthly_panel.csv    CPI, all items, % change on a year earlier
  OECD_ipi_monthly_panel.csv        Industrial production, index, cal. + seas. adj.

The CPI *index* is the primitive - inflation at any horizon can be derived from
it, but the level cannot be recovered from a growth rate, so the index is what
downstream work should build on. The year-on-year panel is kept because it is
the figure the OECD publishes and is convenient for plotting.
"""

import io
import os
from datetime import date

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

REQUEST_TIMEOUT = 180


# ======================================================
# SERIES DEFINITIONS
# ======================================================
#
# CPI key    : REF_AREA.FREQ.METHODOLOGY.MEASURE.UNIT_MEASURE.EXPENDITURE.
#              ADJUSTMENT.TRANSFORMATION
# IPI key    : REF_AREA.FREQ.MEASURE + wildcards, filtered explicitly below
#              (the dataflow's dimension order is not stable enough to pin
#              positionally, so the slice is selected on named columns).
#
# Choices and why:
#   * National CPI methodology (N), not HICP - HICP does not cover the US, so
#     the national measure is the only definition consistent across all four.
#   * CPI not seasonally adjusted - a seasonally adjusted index is not
#     published for the UK, so NSA is the only consistent choice. Seasonally
#     adjust downstream if the application needs it.
#   * All items (_T) - the headline basket.
#   * Industry except construction (BTE) - the standard industrial-production
#     aggregate and the usual monthly output proxy. Construction is excluded
#     because it is driven by a different cycle and is not covered uniformly.
#   * Calendar and seasonally adjusted (Y) for IPI - the published form; the
#     unadjusted series is dominated by trading-day and seasonal effects.

CPI_FLOW = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,"
IPI_FLOW = "OECD.SDD.STES,DSD_STES@DF_INDSERV,"

CPI_INDEX_KEY = "{countries}.M.N.CPI.IX._T.N._Z"
CPI_YOY_KEY = "{countries}.M.N.CPI.PA._T.N.GY"
IPI_KEY = "{countries}.M.PRVM......"

IPI_FILTER = {
    "ACTIVITY": "BTE",       # Industry except construction
    "ADJUSTMENT": "Y",       # Calendar and seasonally adjusted
    "UNIT_MEASURE": "IX"     # Index
}

SERIES = [
    {
        "name": "CPI index (all items, NSA)",
        "flow": CPI_FLOW,
        "key": CPI_INDEX_KEY,
        "filter": {},
        "file": "OECD_cpi_index_monthly_panel.csv"
    },
    {
        "name": "CPI year-on-year (%)",
        "flow": CPI_FLOW,
        "key": CPI_YOY_KEY,
        "filter": {},
        "file": "OECD_cpi_yoy_monthly_panel.csv"
    },
    {
        "name": "Industrial production index (BTE, cal. + seas. adj.)",
        "flow": IPI_FLOW,
        "key": IPI_KEY,
        "filter": IPI_FILTER,
        "file": "OECD_ipi_monthly_panel.csv"
    }
]


# ======================================================
# FETCH
# ======================================================

def fetch(flow, key, selection):
    """Download one slice and pivot it to a country panel.

    Raises if the requested slice is not unique per country-month - an
    ambiguous key must never be resolved by silently taking the first row.
    """

    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        f"{flow}/"
        f"{key.format(countries='+'.join(COUNTRIES))}?"
        f"startPeriod={START_DATE}"
        f"&endPeriod={END_DATE}"
        "&format=csvfilewithlabels"
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text)
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

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # Stable column order, and a hard check that every country came back.
    missing = set(COUNTRIES) - set(panel.columns)

    if missing:
        raise ValueError(f"No data returned for: {sorted(missing)}")

    panel = panel[sorted(COUNTRIES)]
    panel.columns.name = None

    return panel


def report(panel, name):
    """Print the coverage facts worth knowing before the data is used."""

    expected = pd.date_range(
        start=f"{START_DATE}-01",
        end=panel.index.max(),
        freq="MS"
    )

    gaps = expected.difference(panel.index)

    print(f"\n{name}")
    print(f"  span      : {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}"
          f"  ({len(panel)} months)")
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

for spec in SERIES:

    print(f"\nFetching {spec['name']} ...")

    panel = fetch(
        spec["flow"],
        spec["key"],
        spec["filter"]
    )

    report(panel, spec["name"])

    output_path = os.path.join(
        OUTPUT_FOLDER,
        spec["file"]
    )

    panel.to_csv(output_path)

    print("  saved     :", output_path)


print(f"\nFinished. Vintage: {date.today():%Y-%m-%d} "
      "(OECD revises past observations - record this with any results).")
