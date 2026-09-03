library(gasmodel)
library(yaml)

config <- yaml::read_yaml("config.yml")

dir.create(config$results_path, showWarnings = FALSE, recursive = TRUE)

results <- list()

total <- length(config$indicators) * length(config$countries)
counter <- 0

for (dataset in config$indicators) {
  
  dat <- read.csv(config$data_paths[[dataset]],
                  stringsAsFactors = FALSE,
                  check.names = FALSE)
  
  dat$TIME_PERIOD <- as.Date(dat$TIME_PERIOD)
  
  for (country in config$countries) {
    
    counter <- counter + 1
    cat(sprintf("\n[%d/%d] %s - %s\n", counter, total, dataset, country))
    
    forecast_date <- as.Date(config$forecast_start_date)
    
    train <- dat[dat$TIME_PERIOD < forecast_date, ]
    
    y <- train[[country]]
    y <- y[!is.na(y)]
    y <- tail(y, config$lookback)
    
    candidates <- list()
    
    for (distr in c("normal", "t")) {
      for (p in 1:2) {
        for (q in 1:2) {
          
          fit <- tryCatch(
            gas(
              y = y,
              distr = distr,
              param = "meanvar",
              scaling = "unit",
              regress = "joint",
              p = p,
              q = q
            ),
            error = function(e) NULL
          )
          
          if (!is.null(fit)) {
            candidates[[length(candidates) + 1]] <- list(
              model = fit,
              distribution = distr,
              p = p,
              q = q,
              BIC = as.numeric(fit$fit$bic)
            )
          }
        }
      }
    }
    
    if (length(candidates) == 0)
      stop(paste("No GAS model estimated for", dataset, country))
    
    best <- candidates[[which.min(
      sapply(candidates, `[[`, "BIC")
    )]]
    
    cat(sprintf(
      "Selected: %s GAS(%d,%d), BIC = %.2f\n",
      best$distribution,
      best$p,
      best$q,
      best$BIC
    ))
    
    forecast <- gas_forecast(
      gas_object = best$model,
      method = "simulated_paths",
      t_ahead = max(config$forecast_horizons),
      rep_ahead = config$n_samples,
      quant = config$quantiles
    )
    
    for (h in config$forecast_horizons) {
      
      mean_h <- as.numeric(
        forecast$forecast$y_ahead_mean
      )[h]
      
      quant_h <- forecast$forecast$y_ahead_quant
      
      if (length(dim(quant_h)) == 3) {
        quant_h <- quant_h[h, 1, ]
      } else if (length(dim(quant_h)) == 2) {
        if (nrow(quant_h) == length(config$quantiles))
          quant_h <- quant_h[, h]
        else
          quant_h <- quant_h[h, ]
      }
      
      row <- data.frame(
        dataset = dataset,
        country = country,
        date = seq(
          forecast_date,
          by = "month",
          length.out = max(config$forecast_horizons)
        )[h],
        horizon = h,
        mean = mean_h
      )
      
      for (i in seq_along(config$quantiles)) {
        row[[paste0(
          "q",
          sprintf("%02d", config$quantiles[i] * 100)
        )]] <- quant_h[i]
      }
      
      results[[length(results) + 1]] <- row
    }
  }
}

results <- do.call(rbind, results)

write.csv(
  results,
  file.path(config$results_path, "gas.csv"),
  row.names = FALSE
)

print(results)
