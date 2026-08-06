# Data

Monthly panels for Germany, France, the United Kingdom and the United States,
1990-01 to 2025-12, pulled from the OECD SDMX API by `../oecd_api.py`.

**Vintage: 2026-08-06.** The OECD revises past observations, so results are only
reproducible against a stated vintage. Re-running `oecd_api.py` overwrites these
files with the current vintage — record the date alongside any estimates.

## Files

| File | Series | Unit |
|---|---|---|
| `OECD_cpi_index_monthly_panel.csv` | CPI, all items | index |
| `OECD_cpi_yoy_monthly_panel.csv` | CPI, all items | % change on same month a year earlier |
| `OECD_ipi_monthly_panel.csv` | Industrial production, industry except construction | index |

Layout: `TIME_PERIOD` (month start, `YYYY-MM-01`) in column 1, then one column
per country in ISO-3 code (`DEU`, `FRA`, `GBR`, `USA`).

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
