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
* ``OECD_cpi_yoy_latest_monthly_panel.csv``
* ``OECD_ipi_yoy_latest_monthly_panel.csv``

The first two files retain every edition and are the source for real-time
analysis. The two panel files select the latest available release for each
country-month and use the same wide layout as the model input panels:
``TIME_PERIOD`` followed by ISO-3 country columns.

Rebased CPI series drop out of the revisions flow: Japanese CPI ends in
2021-06 from edition 202207 onward, French CPI in 2025-12 from edition 202603
onward. These months are filled from the current COICOP 2018 prices flow. For every edition, each missing month up to
the usual two-month publication lag gets the current value, tagged
``source = "prices_c2018"`` (all other rows are ``"revisions"``). Unadjusted
CPI is rarely revised, so this approximates the real-time series.

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
END_DATE = "2026-08"
OUTPUT_FOLDER = "data"
REQUEST_TIMEOUT = 180

REVISION_FLOW = (
    "OECD.SDD.STES,"
    "DSD_STES_REVISIONS@DF_STES_REVISIONS,"
    "4.0"
)

PRICES_FLOW = (
    "OECD.SDD.TPS,"
    "DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL,"
    "1.0"
)

# Target/country pairs missing from the revisions flow, filled from the prices
# flow: (variable, country) -> (series key, publication lag in months).
SUPPLEMENTS = {
    ("CPI_YOY", "JPN"): ("JPN.M.N.CPI.IX._T.N._Z", 2),
    ("CPI_YOY", "FRA"): ("FRA.M.N.CPI.IX._T.N._Z", 2),
}

OUTPUT_FILES = {
    "CPI_YOY": "OECD_cpi_yoy_vintages_monthly_long.csv",
    "IPI_YOY": "OECD_ipi_yoy_vintages_monthly_long.csv",
}
LATEST_RELEASE_FILES = {
    "CPI_YOY": "OECD_cpi_yoy_latest_monthly_panel.csv",
    "IPI_YOY": "OECD_ipi_yoy_latest_monthly_panel.csv",
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


def fetch_current_index(key):
    """Download one current-vintage monthly index from the prices flow."""

    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        f"{PRICES_FLOW}/{key}?"
        f"startPeriod={START_DATE}"
        f"&endPeriod={END_DATE}"
        "&dimensionAtObservation=AllDimensions"
        "&format=csvfilewithlabels"
    )
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))

    if frame.empty:
        raise ValueError(f"No prices data returned for {key}.")

    index = pd.Series(
        pd.to_numeric(frame["OBS_VALUE"], errors="coerce").values,
        index=pd.to_datetime(frame["TIME_PERIOD"]),
    ).sort_index()
    return index.dropna()


def supplement_vintages(vintages, variable, country, index, lag):
    """Add current YoY values for months an edition is missing.

    Each edition receives the months after its last revisions-flow
    observation, up to ``lag`` months before the edition date.
    """

    yoy = index.pct_change(12).mul(100).dropna()
    target = vintages[
        (vintages["variable"] == variable)
        & (vintages["country"] == country)
    ]
    last_observed = target.groupby("edition")["time_period"].max()
    editions = (
        vintages.loc[vintages["variable"] == variable,
                     ["edition", "edition_period"]]
        .drop_duplicates()
    )

    rows = []
    for edition, edition_period in editions.itertuples(index=False):
        start = last_observed.get(edition, pd.Timestamp.min)
        end = edition_period - pd.DateOffset(months=lag)
        months = yoy[(yoy.index > start) & (yoy.index <= end)]
        rows.append(pd.DataFrame({
            "country": country,
            "time_period": months.index,
            "edition": edition,
            "edition_period": edition_period,
            "value": months.values,
            "variable": variable,
            "source": "prices_c2018",
        }))

    added = pd.concat(rows, ignore_index=True)
    print(f"{variable} {country}: {len(added):,} rows added from prices flow")
    return pd.concat([vintages, added], ignore_index=True)


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
    result["source"] = "revisions"

    return result[[
        "country",
        "time_period",
        "edition",
        "edition_period",
        "value",
        "variable",
        "source",
    ]].sort_values(["country", "time_period", "edition"])


def latest_release_panel(vintages, variable):
    """Return the latest-edition value per country-month in wide format."""

    latest = (
        vintages[vintages["variable"] == variable]
        .sort_values(["country", "time_period", "edition"])
        .groupby(["country", "time_period"], as_index=False)
        .last()
    )
    panel = latest.pivot(
        index="time_period",
        columns="country",
        values="value"
    ).sort_index()
    panel = panel.reindex(columns=COUNTRIES)
    # Fill interior gaps linearly (e.g. USA CPI 2025-10, missing in the OECD
    # data); leave the ragged edge at the end as NaN.
    panel = panel.interpolate(limit_area="inside")
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

    for (variable, country), (key, lag) in SUPPLEMENTS.items():
        print(f"Downloading current {variable} for {country}...")
        vintages = supplement_vintages(
            vintages, variable, country, fetch_current_index(key), lag
        )
    vintages = vintages.sort_values(
        ["variable", "country", "time_period", "edition"]
    )

    for variable, filename in OUTPUT_FILES.items():
        target = vintages[vintages["variable"] == variable].copy()
        target.to_csv(
            os.path.join(OUTPUT_FOLDER, filename),
            index=False
        )
        latest_release_panel(target, variable).to_csv(
            os.path.join(OUTPUT_FOLDER, LATEST_RELEASE_FILES[variable])
        )
        report(vintages, variable)

    print("Finished. Latest-release panels are ready for downstream models.")


if __name__ == "__main__":
    main()
