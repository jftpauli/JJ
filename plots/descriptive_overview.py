"""Simple descriptive plots for the OECD inflation and GDP panels.

Each figure shows all countries in the sample on one panel, so the time-series
shape of the series can be read at a glance.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plots.style import COUNTRY_LABELS, apply_rcparams, style_axes

DATA_FOLDER = os.path.join(ROOT, "data")
FIGURE_FOLDER = os.path.join(ROOT, "figures")


def load_panel(filename):
    """Load one OECD panel CSV and sort it by date."""

    path = os.path.join(DATA_FOLDER, filename)
    panel = pd.read_csv(path, index_col="TIME_PERIOD", parse_dates=True)
    panel = panel.sort_index()
    return panel


def plot_panel(panel, title, ylabel, output_name):
    """Draw a single-line-per-country time series plot."""

    apply_rcparams(tick_size=9)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for country in panel.columns:
        ax.plot(
            panel.index,
            panel[country],
            label=COUNTRY_LABELS.get(country, country),
            linewidth=1.8,
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="upper right", ncol=min(4, len(panel.columns)))
    style_axes(ax)

    fig.tight_layout()
    path = os.path.join(FIGURE_FOLDER, output_name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def main():
    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    inflation = load_panel("OECD_cpi_yoy_monthly_panel.csv")
    gdp = load_panel("OECD_gdp_growth_quarterly_panel.csv")

    plot_panel(
        inflation,
        "OECD CPI inflation, year-on-year (%)",
        "Inflation (% y/y)",
        "cpi_inflation_descriptive.png",
    )

    plot_panel(
        gdp,
        "OECD real GDP growth, quarter-on-quarter (%)",
        "Growth (% q/q)",
        "gdp_growth_descriptive.png",
    )


if __name__ == "__main__":
    main()
