# Score-driven (GAS) quantile forecasts for the OECD CPI/IPI panels.
#
# Not part of run_forecasts.py: gasmodel is R-only, so this runs separately
# and writes results/gas_<indicator>_forecasts.csv in the same layout as the
# Python models (country, origin, quantile columns, horizon, actual) - one
# file per indicator listed in config.toml's INDICATORS.
#
# Rolling-origin backtest, as in models/qar.py: for every origin from
# FORECAST_START_DATE onward, the model sees only data up to and including
# that origin and is asked for both FORECAST_HORIZONS (1 and 12 months
# ahead), stopping once an origin's target postdates the last known actual.
# The GAS specification (distribution, p, q) is chosen once via BIC on the
# initial LOOKBACK window and re-fit at every origin - a full grid search per
# origin would be prohibitively slow for little BIC gain.

library(gasmodel)
library(RcppTOML)

config <- parseTOML("config.toml")

dir.create(config$RESULTS_PATH, showWarnings = FALSE, recursive = TRUE)

forecast_date <- as.Date(config$FORECAST_START_DATE)
horizons <- config$FORECAST_HORIZONS
quantiles <- config$QUANTILES
quantile_cols <- paste0("q", quantiles)

# GAS specifications tried for the initial fit; the lowest-BIC one is reused
# for every origin. Kept deliberately small - each combination needs a full
# simulation-based forecast, and the panels are short monthly series.
DISTRIBUTIONS <- c("normal", "t")
ORDERS <- 1:2

#' Add `n` months to a Date, respecting month-end rollover.
add_months <- function(date, n) seq(date, by = "month", length.out = n + 1)[n + 1]

#' Fit every (distribution, p, q) combination on `y`, return the lowest-BIC fit
#' (or NULL if none converged).
select_gas_spec <- function(y) {
  candidates <- list()

  for (distr in DISTRIBUTIONS) {
    for (p in ORDERS) {
      for (q in ORDERS) {
        fit <- tryCatch(
          gas(y = y, distr = distr, param = "meanvar", scaling = "unit",
              regress = "joint", p = p, q = q),
          error = function(e) NULL
        )
        if (!is.null(fit)) {
          candidates[[length(candidates) + 1]] <- list(
            distribution = distr, p = p, q = q, bic = as.numeric(fit$fit$bic)
          )
        }
      }
    }
  }

  if (length(candidates) == 0) return(NULL)
  candidates[[which.min(sapply(candidates, `[[`, "bic"))]]
}

#' Fit `spec` on `window` and simulate `max(horizons)` steps ahead, returning
#' one output row per horizon with mean, quantile columns and the realised
#' value at that target date.
forecast_rows <- function(spec, window, origin, country, full_series, last_date) {

  fit <- tryCatch(
    gas(y = window, distr = spec$distribution, param = "meanvar",
        scaling = "unit", regress = "joint", p = spec$p, q = spec$q),
    error = function(e) NULL
  )
  if (is.null(fit)) return(NULL)

  sim <- gas_forecast(
    gas_object = fit,
    method = "simulated_paths",
    t_ahead = max(horizons),
    rep_ahead = config$N_SAMPLES,
    quant = quantiles
  )

  rows <- lapply(horizons, function(h) {

    target_date <- add_months(origin, h)
    if (target_date > last_date) return(NULL)  # ran out of actuals to compare against

    quant_h <- sim$forecast$y_ahead_quant
    # Array shape depends on rep_ahead/quant settings; normalise to a plain
    # vector of quantiles for horizon h.
    if (length(dim(quant_h)) == 3) {
      quant_h <- quant_h[h, 1, ]
    } else if (nrow(quant_h) == length(quantiles)) {
      quant_h <- quant_h[, h]
    } else {
      quant_h <- quant_h[h, ]
    }

    # `[` (not `[[`) on a named vector returns NA_real_, rather than erroring,
    # when the target date is not among the realised observations.
    actual <- unname(full_series[as.character(target_date)])

    row <- data.frame(
      country = country,
      origin = origin,
      horizon = h,
      mean = as.numeric(sim$forecast$y_ahead_mean)[h]
    )
    row[quantile_cols] <- as.list(quant_h)
    row$actual <- actual
    row
  })

  do.call(rbind, rows)
}

for (dataset in config$INDICATORS) {

  path <- config$DATA_PATHS[[dataset]]
  if (is.null(path)) next  # no data configured for this indicator, skip

  dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  dat$TIME_PERIOD <- as.Date(dat$TIME_PERIOD)

  results <- list()

  for (country in config$COUNTRIES) {

    series <- dat[!is.na(dat[[country]]), c("TIME_PERIOD", country)]
    names(series) <- c("date", "value")
    if (nrow(series) == 0) next

    last_date <- max(series$date)

    # Realised values keyed by date string, used to fill `actual` for target
    # dates that have since been observed.
    full_series <- setNames(series$value, as.character(series$date))

    init_window <- tail(series$value[series$date < forecast_date], config$LOOKBACK)
    spec <- select_gas_spec(init_window)
    if (is.null(spec)) {
      warning(sprintf("No GAS model estimated for %s - %s", dataset, country))
      next
    }

    cat(sprintf("%s - %s: selected %s GAS(%d,%d), BIC = %.2f\n",
                dataset, country, spec$distribution, spec$p, spec$q, spec$bic))

    origins <- series$date[series$date >= forecast_date & series$date <= last_date]

    for (origin in origins) {
      origin <- as.Date(origin, origin = "1970-01-01")

      if (add_months(origin, min(horizons)) > last_date) break  # no origin from here on has an actual to compare against

      window <- tail(series$value[series$date <= origin], config$LOOKBACK)
      rows <- forecast_rows(spec, window, origin, country, full_series, last_date)
      if (!is.null(rows)) results[[length(results) + 1]] <- rows
    }
  }

  out <- do.call(rbind, results)
  out_path <- file.path(config$RESULTS_PATH, sprintf("gas_%s_forecasts.csv", dataset))
  write.csv(out, out_path, row.names = FALSE)
  cat(sprintf("saved %s\n", out_path))
}

