# Data

The OECD vintage workflow is implemented in `../oecd_api.py`. It downloads
CPI and industrial-production index observations from the OECD revisions
dataflow, derives 12-month percentage growth within each OECD edition, and
writes only the YoY targets to disk. Raw index values are used in memory only.

## Outputs

| File | Contents |
|---|---|
| `OECD_cpi_yoy_vintages_monthly_long.csv` | All CPI YoY observations by country, period, and OECD edition |
| `OECD_ipi_yoy_vintages_monthly_long.csv` | All IPI YoY observations by country, period, and OECD edition |
| `OECD_cpi_yoy_latest_monthly_panel.csv` | Latest CPI release per country-month, wide model-input layout |
| `OECD_ipi_yoy_latest_monthly_panel.csv` | Latest IPI release per country-month, wide model-input layout |

The long files contain `country`, `time_period`, `edition`,
`edition_period`, `value`, `variable`, and `source`. `edition` is the OECD publication
vintage (`YYYYMM`). The panel files contain `TIME_PERIOD` followed by the
country columns `DEU`, `FRA`, `GBR`, `USA`, and `JPN`, matching the layout
expected by the forecasting code.

The two panel files select the latest edition containing each observation
(the fully revised series). First releases can be recovered from the long
files by taking the earliest edition per country-month. For a genuine real-time forecast at
origin $t$, select the latest edition available by $t$ from the corresponding
long vintage file rather than using the first-release panel.

CPI and IPI YoY values are calculated as:

```text
100 * (index[t] / index[t - 12] - 1)
```

The transformation is performed separately within each country and edition, so
it never combines observations from different vintages. The revisions flow
does not expose a separate IPI adjustment dimension; the selected IPI series
is `MEASURE=PRVM`, `UNIT_MEASURE=IX`, `ACTIVITY=BTE`.

Rebased CPI series drop out of the revisions flow: Japan ends in 2021-06 from
edition 202207 onward, France in 2025-12 from edition 202603 onward. Those
months are filled from the
current OECD COICOP 2018 prices flow, with a two-month publication lag per
edition, and are marked `source = prices_c2018`. These rows are an
approximation of the real-time data, not true vintages; all other rows are
`source = revisions`.
