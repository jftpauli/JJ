"""Fetch the Chicago Fed National Financial Conditions Index from FRED.

NFCI is the conditioning variable of the growth-at-risk literature. Adrian,
Boyarchenko & Giannone (2019) regress future average GDP growth on it at
quarterly frequency; Loria, Matthes & Zhang (2025) do the same for industrial
production at monthly frequency. It carries almost all of the left-tail
movement in those papers, so a growth-at-risk exercise without it is a
different exercise.

Written to data/:

  FRED_nfci_monthly.csv    NFCI, monthly average of the weekly index

Two things about this series that matter downstream:

* **It is published weekly, on Fridays.** The monthly value here is the mean of
  the weekly observations dated inside the month - the same convention as
  FRED's own monthly aggregation of NFCI, and what the papers above use. Taking
  the last week of the month instead would be defensible but noisier; taking
  the first would discard information available at the origin.
* **It is US-only.** There is no NFCI for Germany, France or the UK, so the
  models that condition on it can only be run on the US column of a panel. The
  euro-area analogue would be the ECB's CISS, which is a different index with a
  different construction, not a like-for-like substitute.

Revisions: the NFCI is re-estimated as its input series are revised, so this is
a *current-vintage* series used as if it had been available in real time. That
is the same assumption the papers make, and it is an assumption, not a fact.
"""

import io
import os
from datetime import date

import pandas as pd
import requests


# ======================================================
# USER SETTINGS
# ======================================================

SERIES_ID = "NFCI"

START_DATE = "1990-01"
END_DATE = "2025-12"

OUTPUT_FOLDER = "data"
OUTPUT_FILE = "FRED_nfci_monthly.csv"

REQUEST_TIMEOUT = 120


def fetch(series_id=SERIES_ID):
    """Download one FRED series as a dated float column.

    The CSV endpoint needs no API key. Missing observations arrive as "." and
    are coerced to NaN rather than silently dropped, so a gap stays visible.
    """

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
    )

    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    frame = pd.read_csv(
        io.StringIO(response.text),
        parse_dates=["observation_date"],
        index_col="observation_date"
    )

    if series_id not in frame.columns:
        raise ValueError(
            f"FRED returned columns {list(frame.columns)}, expected {series_id}"
        )

    return pd.to_numeric(frame[series_id], errors="coerce").sort_index()


def to_monthly(weekly):
    """Monthly average of a weekly series, indexed at the month start.

    A month is only reported if every week dated inside it is observed; a
    partial month at the end of the sample would otherwise enter the panel as a
    full observation computed from one or two weeks.
    """

    grouped = weekly.groupby(pd.Grouper(freq="MS"))

    monthly = grouped.mean()
    complete = grouped.apply(lambda g: g.notna().all() and len(g) > 0)

    return monthly.where(complete)


def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Fetching {SERIES_ID} from FRED ...")

    weekly = fetch()
    monthly = to_monthly(weekly).loc[START_DATE:END_DATE]

    panel = monthly.to_frame(name="USA")
    panel.index.name = "TIME_PERIOD"

    print(f"\n{SERIES_ID} (monthly average of weekly)")
    print(f"  span      : {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}"
          f"  ({len(panel)} months)")
    print(f"  missing   : {int(panel['USA'].isna().sum())}")
    print(f"  last obs  : {panel['USA'].iloc[-1]:.3f}")
    print(f"  range     : {panel['USA'].min():.2f} to {panel['USA'].max():.2f}")

    path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    panel.round(6).to_csv(path)

    print("  saved     :", path)

    print(f"\nFinished. Vintage: {date.today():%Y-%m-%d} "
          "(the NFCI is re-estimated as its inputs are revised - record this "
          "with any results).")


if __name__ == "__main__":
    main()
