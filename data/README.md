# Data

Monthly panels for Germany, France, the United Kingdom and the United States,
1990-01 to 2025-12, pulled from the OECD SDMX API by `../oecd_api.py`.

**Vintage: 2026-08-06.** The OECD revises past observations, so results are only
reproducible against a stated vintage. Re-running `oecd_api.py` overwrites these
files with the current vintage — record the date alongside any estimates.

## Files

| File | Series | Unit | Fetched or derived |
|---|---|---|---|
| `OECD_cpi_index_monthly_panel.csv` | CPI, all items | index | fetched |
| `OECD_cpi_yoy_monthly_panel.csv` | CPI, all items | % change on same month a year earlier | fetched |
| `OECD_ipi_monthly_panel.csv` | Industrial production, industry except construction | index | fetched |
| `OECD_ipi_growth_monthly_panel.csv` | Industrial production, industry except construction | monthly %, `100 × Δlog` | derived |
| `OECD_ipi_growth12_monthly_panel.csv` | Industrial production, industry except construction | 12-month %, `100 × Δ₁₂log` | derived |
| `FRED_nfci_monthly.csv` | Chicago Fed National Financial Conditions Index | index, monthly average of weekly | fetched (`../fred_api.py`) |

Layout: `TIME_PERIOD` (month start, `YYYY-MM-01`) in column 1, then one column
per country in ISO-3 code (`DEU`, `FRA`, `GBR`, `USA`).

**The two `_yoy_` / `_growth_` panels are the model-ready ones.** They are
already stationary transforms, so forecasting code loads them as they stand and
performs no transformation of its own — there is exactly one definition of each
target, and it lives in `oecd_api.py`.

The index panels are kept because a level cannot be recovered from a growth
rate. They are the primitive any later change of transformation starts from.

`OECD_ipi_growth_monthly_panel.csv` is derived from the index in the same run
rather than fetched separately, which guarantees the two are internally
consistent. The log difference is used rather than a simple percentage change:
log differences are additive across periods and symmetric in sign, which is what
a forecasting target should be. It starts 1990-02 (431 months) — one month
shorter than the index, since the first period has nothing to difference
against.

`OECD_ipi_growth12_monthly_panel.csv` is the **growth-at-risk target**: the same
index, differenced over twelve months instead of one, starting 1991-01 (420
months). Up to the 1/12 scaling it is the "average growth over the next year"
that Adrian, Boyarchenko & Giannone (2019, *AER*) forecast for GDP and Loria,
Matthes & Zhang (2025, *EJ*) for industrial production — averaging twelve
consecutive monthly log changes telescopes to the twelve-month log change.

Two properties of this panel that change how it must be handled:

- **Observations overlap.** Consecutive months share eleven of their twelve
  months, so the series is strongly autocorrelated by construction and every
  standard error computed on it has to be HAC, or computed on origins thinned
  to every twelfth month. This is intrinsic to the target, not a defect.
- **It is far smoother than the monthly panel** — an order of magnitude more
  persistent — and the two must not be compared on the same scale or read as
  the same forecasting problem. Monthly IP growth is close to unforecastable;
  the twelve-month rate is not.

`FRED_nfci_monthly.csv` is the conditioning variable of that literature, and the
only series here that is **not from the OECD and not a panel** — the NFCI is a
US index, so the file has a single `USA` column and the models that condition on
it run only on the US. FRED publishes it weekly (Fridays); the monthly value is
the mean of the weeks dated inside the month, and a month is only reported if
every one of its weeks is observed. The ECB's CISS is the nearest euro-area
analogue but is a differently constructed index, not a drop-in substitute.

Both the NFCI and the OECD panels are **current-vintage series used as though
they had been available in real time**. The NFCI in particular is re-estimated
as its inputs are revised. The papers above make the same assumption; it is
still an assumption, and a genuinely real-time exercise would need vintage data
(ALFRED for the US).

## Series keys

CPI — dataflow `OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL`:

```
index : {country}.M.N.CPI.IX._T.N._Z
yoy   : {country}.M.N.CPI.PA._T.N.GY
```

IPI — dataflow `OECD.SDD.STES,DSD_STES@DF_INDSERV`, measure `PRVM`, selected on
`ACTIVITY=BTE`, `ADJUSTMENT=Y`, `UNIT_MEASURE=IX`.

## Definitional choices

- **National CPI methodology, not HICP.** HICP does not cover the United States,
  so the national measure is the only definition available for all four countries.
  Note this is a genuine cross-country inconsistency in *concept* — the national
  CPIs differ in owner-occupied housing treatment in particular (the US CPI uses
  owners' equivalent rent; the UK CPI excludes owner-occupier costs).
- **CPI not seasonally adjusted.** A seasonally adjusted index is not published
  for the UK, so NSA is the only consistent choice. Seasonally adjust downstream
  if needed.
- **Industry except construction (`BTE`)** for IPI — the standard industrial
  production aggregate and the usual monthly output proxy. Construction is
  excluded: it follows a different cycle and is not covered uniformly.
- **Calendar and seasonally adjusted (`Y`)** for IPI — the published form. The
  unadjusted series is dominated by trading-day and seasonal effects.
- **Index bases are not harmonised across countries.** Levels are not comparable
  cross-sectionally; use growth rates, or re-base to a common period, before
  comparing across countries.

## Known data issues

- **US CPI is missing 2025-10** in both CPI panels (the OECD has no observation;
  it is not a pipeline bug). IPI is complete, with no gaps in any country.
- **UK year-on-year cannot be cleanly derived from the UK index.** The UK CPI
  index is published to one decimal, so deriving YoY from it differs from the
  published rate by up to 0.18pp. For DEU/FRA/USA the two agree to 0.000pp. Use
  the published `..._yoy_...` panel for the UK rather than differencing the index.
- **Do not query these dataflows with wildcard keys.** The IPI dataflow returns
  several economic activities (industry, manufacturing, electricity,
  construction) and both adjustment modes for the same country-month. Collapsing
  that with `pivot_table(aggfunc="first")` silently selects an arbitrary one per
  country — and can select a *different* one in different months, producing a
  column that is not a single series. `oecd_api.py` raises if a key fails to
  resolve to one observation per country-month.
