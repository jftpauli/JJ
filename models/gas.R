# GAS forecasts - step 1: read settings from config.toml
library(RcppTOML)

# Set this to the project root (the folder containing config.toml)
project_root <- "C:/Users/juliu/Documents/Forschung/JJ"

config <- parseTOML(file.path(project_root, "config.toml"))

countries           <- config$COUNTRIES
indicators          <- config$INDICATORS
lookback            <- config$LOOKBACK
n_samples           <- config$N_SAMPLES
horizons            <- config$FORECAST_HORIZONS
quantiles           <- config$QUANTILES
forecast_start_date <- as.Date(config$FORECAST_START_DATE)
results_path        <- file.path(project_root, config$RESULTS_PATH)
data_paths          <- lapply(config$DATA_PATHS, function(p) file.path(project_root, p))


# GAS step 2: AIC model selection per indicator x country (no forecasting)
library(gasmodel)

distributions <- c("norm", "t")
orders        <- 1:2  # score order p and autoregressive order q

aic_table <- NULL
n_fits <- length(indicators) * length(countries) * length(distributions) * (length(orders)^2 - 1)
done   <- 0
for (indicator in indicators) {

  panel <- read.csv(data_paths[[indicator]], check.names = FALSE)
  panel$TIME_PERIOD <- as.Date(panel$TIME_PERIOD)

  for (country in countries) {

    # Selection uses only data before the first forecast origin
    y <- panel[[country]][panel$TIME_PERIOD < forecast_start_date]
    y <- y[!is.na(y)]

    for (distr in distributions) {
      for (p in orders) {
        for (q in orders) {
          if (p == 2 && q == 2) next   # excluded specification

          fit <- tryCatch(
            gas(y = y, distr = distr, p = p, q = q),
            error = function(e) NULL
          )

          aic_table <- rbind(aic_table, data.frame(
            indicator    = indicator,
            country      = country,
            distribution = distr,
            p            = p,
            q            = q,
            aic          = if (is.null(fit)) NA_real_ else as.numeric(fit$fit$aic),
            stringsAsFactors = FALSE
          ))
          done <- done + 1
          cat(sprintf("\rAIC fits: %d/%d", done, n_fits))
          if (done == n_fits) cat("\n")
        }
      }
    }
  }
}

# Best specification (lowest AIC) for each time series
key  <- paste(aic_table$indicator, aic_table$country)
best <- do.call(rbind, lapply(split(aic_table, key), function(d) {
  d[which.min(d$aic), ]
}))
rownames(best) <- NULL

# Full ranking per series, then the winners
aic_table <- aic_table[order(aic_table$indicator, aic_table$country, aic_table$aic), ]
print(aic_table, row.names = FALSE)
cat("\nBest specification per series (lowest AIC):\n")
print(best, row.names = FALSE)


# GAS step 3: rolling-origin forecasts with the best specification per series
set.seed(1)
dir.create(results_path, showWarnings = FALSE, recursive = TRUE)

quantile_cols <- paste0("q", quantiles)
max_h <- max(horizons)

# Quantiles of horizon h from gas_forecast's y_ahead_quant, whatever its layout
horizon_quantiles <- function(q_array, h) {
  d <- dim(q_array)
  if (length(d) == 3) q_array <- q_array[, 1, ]   # drop the series dimension
  if (nrow(q_array) == max_h) as.numeric(q_array[h, ]) else as.numeric(q_array[, h])
}

for (indicator in indicators) {

  panel <- read.csv(data_paths[[indicator]], check.names = FALSE)
  panel$TIME_PERIOD <- as.Date(panel$TIME_PERIOD)

  country_results <- list()

  for (country in countries) {

    spec <- best[best$indicator == indicator & best$country == country, ]
    if (nrow(spec) == 0 || is.na(spec$aic)) next

    keep   <- !is.na(panel[[country]])
    dates  <- panel$TIME_PERIOD[keep]
    values <- panel[[country]][keep]
    n      <- length(values)

    origins <- which(dates >= forecast_start_date)

    # One fit and simulation per origin, on the expanding history up to it
    forecasts <- list()
    for (i in origins) {
      if (i + min(horizons) > n) next

      fit <- tryCatch(
        gas(y = values[1:i], distr = spec$distribution, p = spec$p, q = spec$q),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        warning(sprintf("%s %s: fit failed at %s", indicator, country, dates[i]))
        next
      }

      sim <- gas_forecast(
        gas_object = fit,
        method     = "simulated_paths",
        t_ahead    = max_h,
        rep_ahead  = n_samples,
        quant      = quantiles
      )
      forecasts[[as.character(i)]] <- sim$forecast$y_ahead_quant
    }

    # Same layout as the Python models: all origins for one horizon, then the next
    rows <- list()
    for (h in horizons) {
      for (i in origins) {
        key_i <- as.character(i)
        if (i + h > n || is.null(forecasts[[key_i]])) next

        row <- data.frame(country = country, origin = dates[i])
        row[quantile_cols] <- as.list(horizon_quantiles(forecasts[[key_i]], h))
        row$horizon <- h
        row$actual  <- values[i + h]
        rows[[length(rows) + 1]] <- row
      }
    }
    if (length(rows) > 0) country_results[[country]] <- do.call(rbind, rows)
    cat(sprintf("%s %s done\n", indicator, country))
  }

  out <- do.call(rbind, country_results)
  out_path <- file.path(results_path, sprintf("gas_%s_forecasts.csv", indicator))
  write.csv(out, out_path, row.names = FALSE)
  cat("saved", out_path, "\n")
}
