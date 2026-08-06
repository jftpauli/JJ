import io
import os
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

GET_CPI = True
GET_IPI = True

OUTPUT_FOLDER = "data"
OUTPUT_FILE_CPI = "OECD_cpi_monthly_panel.csv"
OUTPUT_FILE_IPI = "OECD_ipi_monthly_panel.csv"


# ======================================================
# CREATE OUTPUT FOLDER
# ======================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ======================================================
# CPI
# ======================================================

if GET_CPI:

    countries_api = "+".join(COUNTRIES)

    cpi_url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,/"
        f"{countries_api}.M.N.CPI.PA._T.N.GY?"
        f"startPeriod={START_DATE}"
        f"&endPeriod={END_DATE}"
        "&format=csvfilewithlabels"
    )

    print("Fetching CPI data...")

    response_cpi = requests.get(cpi_url)

    df_cpi = pd.read_csv(
        io.StringIO(response_cpi.text)
    )


    pivot_cpi = df_cpi.pivot_table(
        index="TIME_PERIOD",
        columns="REF_AREA",
        values="OBS_VALUE",
        aggfunc="first"
    )

    pivot_cpi.index = pd.to_datetime(
        pivot_cpi.index
    )

    pivot_cpi = pivot_cpi.sort_index()

    output_path_cpi = os.path.join(
        OUTPUT_FOLDER,
        OUTPUT_FILE_CPI
    )

    pivot_cpi.to_csv(
        output_path_cpi
    )


    print("\nCPI panel:")
    print(pivot_cpi.head())
    print("Saved:", output_path_cpi)



# ======================================================
# INDUSTRIAL PRODUCTION
# ======================================================

if GET_IPI:

    countries_api = "+".join(COUNTRIES)

    ipi_url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.STES,DSD_STES@DF_INDSERV,/"
        f"{countries_api}.M.PRVM......?"
        f"startPeriod={START_DATE}"
        f"&endPeriod={END_DATE}"
        "&format=csvfilewithlabels"
    )

    print("\nFetching Industrial Production data...")

    response_ipi = requests.get(ipi_url)

    df_ipi = pd.read_csv(
        io.StringIO(response_ipi.text)
    )


    pivot_ipi = df_ipi.pivot_table(
        index="TIME_PERIOD",
        columns="REF_AREA",
        values="OBS_VALUE",
        aggfunc="first"
    )

    pivot_ipi.index = pd.to_datetime(
        pivot_ipi.index
    )

    pivot_ipi = pivot_ipi.sort_index()

    output_path_ipi = os.path.join(
        OUTPUT_FOLDER,
        OUTPUT_FILE_IPI
    )

    pivot_ipi.to_csv(
        output_path_ipi
    )


    print("\nIPI panel:")
    print(pivot_ipi.head())
    print("Saved:", output_path_ipi)


print("\nFinished.")