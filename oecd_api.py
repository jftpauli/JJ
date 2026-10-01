"""Download OECD real-time CPI and industrial-production growth vintages.

The OECD revisions dataflow contains historical editions of published
observations. This script keeps only the two model targets:

* CPI all-items year-on-year percentage change (``CPI_YOY``)
* Industrial production, industry except construction, year-on-year
  percentage change (``IPI_YOY``)

The revisions dataflow supplies the underlying index values. YoY growth is
computed separately within each country and OECD edition, so no transformation
mixes values from different vintages. The index values are used in memory only
and are never written to disk.

Outputs in ``data/``:

* ``OECD_cpi_yoy_vintages_monthly_long.csv``
* ``OECD_ipi_yoy_vintages_monthly_long.csv``
* ``OECD_cpi_yoy_monthly_panel.csv``
* ``OECD_ipi_yoy_monthly_panel.csv``

The first two files retain every edition and are the source for real-time
analysis. The two panel files select the first available release for each
country-month and use the same wide layout as the model input panels:
``TIME_PERIOD`` followed by ISO-3 country columns.

The revisions flow does not expose a separate IPI adjustment dimension. The
IPI series is selected by its stable revisions-flow codes (``PRVM``, ``IX``,
``BTE``); its adjustment status should be checked against the OECD metadata
before comparing it with a seasonally adjusted current-vintage series.
"""

import io
import os

import pandas as pd
import requests


COUNTRIES = ["DEU", "FRA", "GBR", "USA", "JPN"]
START_DATE = "1990-01"
END_DATE = "2025-12"
OUTPUT_FOLDER = "data"
REQUEST_TIMEOUT = 180

REVISION_FLOW = (
    "OECD.SDD.STES,"
    "DSD_STES_REVISIONS@DF_STES_REVISIONS,"
    "4.0"
)

OUTPUT_FILES = {
    "CPI_YOY": "OECD_cpi_yoy_vintages_monthly_long.csv",
    "IPI_YOY": "OECD_ipi_yoy_vintages_monthly_long.csv",
}
FIRST_RELEASE_FILES = {
    "CPI_YOY": "OECD_cpi_yoy_monthly_panel.csv",
    "IPI_YOY": "OECD_ipi_yoy_monthly_panel.csv",
}


def fetch_revisions(country):
    """Download all monthly revision rows for one country."""

    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        f"{REVISION_FLOW}/{country}.M....?"
        f"startPeriod={START_DATE}"
        f"&endPeriod={END_DATE}"
        "&dimensionAtObservation=AllDimensions"
        "&format=csvfilewithlabels"
    )
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text), low_memory=False)

    if frame.empty:
        raise ValueError(f"No revisions returned for {country}.")

    return frame


def select_index(frame, variable):
    """Select the exact revisions-flow index underlying one target."""

    if variable == "CPI_YOY":
        mask = (
            (frame["MEASURE"] == "CP")
            & (frame["UNIT_MEASURE"] == "IX")
            & (frame["ACTIVITY"] == "_Z")
        )
    elif variable == "IPI_YOY":
        mask = (
            (frame["MEASURE"] == "PRVM")
            & (frame["UNIT_MEASURE"] == "IX")
            & (frame["ACTIVITY"] == "BTE")
        )
    else:
        raise ValueError(f"Unsupported target: {variable}")

    selected = frame.loc[mask, [
        "TIME_PERIOD", "EDITION", "OBS_VALUE"
    ]].copy()

    if selected.empty:
        raise ValueError(f"No index observations found for {variable}.")

    selected["time_period"] = pd.to_datetime(selected.pop("TIME_PERIOD"))
    selected["edition"] = selected.pop("EDITION").astype(str)
    selected["value"] = pd.to_numeric(
        selected.pop("OBS_VALUE"),
        errors="coerce"
    )
    selected = selected.dropna(subset=["time_period", "value"])
    selected["edition_period"] = pd.PeriodIndex(
        selected["edition"],
        freq="M"
    ).to_timestamp()

    duplicates = selected.duplicated(
        subset=["time_period", "edition"]
    ).sum()
    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate rows found for {variable}; "
            "the revisions selection is ambiguous."
        )

    return selected


def derive_yoy(index_vintages, variable, country):
    """Derive 12-month growth independently inside each OECD edition."""

    result = index_vintages.sort_values(
        ["edition", "time_period"]
    ).copy()
    result["value"] = (
        result.groupby("edition")["value"]
        .pct_change(12)
        .mul(100)
    )
    result = result.dropna(subset=["value"])
    result["country"] = country
    result["variable"] = variable

    return result[[
        "country",
        "time_period",
        "edition",
        "edition_period",
        "value",
        "variable",
    ]].sort_values(["country", "time_period", "edition"])


def first_release_panel(vintages, variable):
    """Return the earliest-edition value per country-month in wide format."""

    first = (
        vintages[vintages["variable"] == variable]
        .sort_values(["country", "time_period", "edition"])
        .groupby(["country", "time_period"], as_index=False)
        .first()
    )
    panel = first.pivot(
        index="time_period",
        columns="country",
        values="value"
    ).sort_index()
    panel = panel.reindex(columns=COUNTRIES)
    panel.columns.name = None
    panel.index.name = "TIME_PERIOD"
    return panel


def report(vintages, variable):
    """Print a compact coverage summary for one target."""

    subset = vintages[vintages["variable"] == variable]
    print(
        f"{variable}: {len(subset):,} rows, "
        f"{subset['edition'].nunique()} editions, "
        f"{subset['time_period'].min():%Y-%m} to "
        f"{subset['time_period'].max():%Y-%m}"
    )


def main():
    """Download, transform, and save the two vintage targets."""

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_vintages = []

    for country in COUNTRIES:
        print(f"Downloading {country}...")
        revisions = fetch_revisions(country)

        for variable in OUTPUT_FILES:
            index = select_index(revisions, variable)
            all_vintages.append(
                derive_yoy(index, variable, country)
            )

    vintages = pd.concat(all_vintages, ignore_index=True)

    for variable, filename in OUTPUT_FILES.items():
        target = vintages[vintages["variable"] == variable].copy()
        target.to_csv(
            os.path.join(OUTPUT_FOLDER, filename),
            index=False
        )
        first_release_panel(target, variable).to_csv(
            os.path.join(OUTPUT_FOLDER, FIRST_RELEASE_FILES[variable])
        )
        report(vintages, variable)

    print("Finished. First-release panels are ready for downstream models.")


if __name__ == "__main__":
    main()
