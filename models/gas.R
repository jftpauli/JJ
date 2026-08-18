# =============================================================================
# ONE-STEP-AHEAD PROBABILISTIC GAS FORECAST
# U.S. CPI YEAR-OVER-YEAR INFLATION
# =============================================================================
#
# Forecast target:
#   2024-01-01
#
# Estimation sample:
#   All observations strictly before 2024-01-01
#
# Model selection:
#   Distribution: normal, Student-t
#   p: 0, 1, 2, 3
#   q: 0, 1, 2, 3
#   Selection criterion: BIC
#
# Forecast:
#   One-step-ahead predictive distribution
#   10,000 simulated paths
#
# =============================================================================


# -----------------------------------------------------------------------------
# 1. PACKAGES
# -----------------------------------------------------------------------------

library(gasmodel)


# -----------------------------------------------------------------------------
# 2. FILE PATHS
# -----------------------------------------------------------------------------

input_file <- "C:/Users/juliu/Documents/Forschung/JJ/data/OECD_cpi_yoy_monthly_panel.csv"

model_dir <- "C:/Users/juliu/Documents/Forschung/JJ/models/"

if (!dir.exists(model_dir)) {
  dir.create(
    model_dir,
    recursive = TRUE
  )
}

forecast_file <- file.path(
  model_dir,
  "GAS_one_step_forecast_USA_CPI_2024_01.csv"
)

selection_file <- file.path(
  model_dir,
  "GAS_one_step_model_selection_USA_CPI_2024_01.csv"
)


# -----------------------------------------------------------------------------
# 3. READ PREPARED DATA
# -----------------------------------------------------------------------------

# The previous script is responsible for:
#   - cleaning,
#   - sorting,
#   - checking monthly completeness,
#   - handling missing observations,
#   - creating the panel.
#
# We therefore do not repeat those operations here.

dat <- read.csv2(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# If the file is comma-separated rather than semicolon-separated,
# read.csv2() will usually produce one column. Re-read in that case.

if (ncol(dat) == 1) {
  
  dat <- read.csv(
    input_file,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}


# -----------------------------------------------------------------------------
# 4. SELECT TIME_PERIOD AND USA
# -----------------------------------------------------------------------------

# Find the two required columns by their imported names.

time_col <- which(
  trimws(names(dat)) == "TIME_PERIOD"
)

usa_col <- which(
  trimws(names(dat)) == "USA"
)

if (length(time_col) != 1 || length(usa_col) != 1) {
  
  cat("\nImported column names:\n")
  dput(names(dat))
  
  stop(
    "Could not find TIME_PERIOD and/or USA."
  )
}

dat <- dat[
  ,
  c(time_col, usa_col),
  drop = FALSE
]

names(dat) <- c(
  "TIME_PERIOD",
  "USA"
)


# -----------------------------------------------------------------------------
# 5. CONVERT DATE
# -----------------------------------------------------------------------------

if (!inherits(dat$TIME_PERIOD, "Date")) {
  
  dat$TIME_PERIOD <- as.Date(
    dat$TIME_PERIOD
  )
}


# -----------------------------------------------------------------------------
# 6. FORECAST SETTINGS
# -----------------------------------------------------------------------------

# Forecast target.

forecast_date <- as.Date(
  "2024-01-01"
)

# Candidate GAS score orders.

candidate_p <- 1:2

# Candidate GAS autoregressive orders.

candidate_q <- 1:2

# Candidate conditional distributions.

candidate_distributions <- c(
  "t"
)

# Distribution parametrization.

candidate_param <- "meanvar"

# GAS score scaling.

scaling <- "unit"

# Joint dynamic specification.

regress <- "joint"

# Number of simulated predictive paths.

n_sim <- 10000L

# Quantiles of the predictive distribution.

forecast_quantiles <- c(
  0.025,
  0.10,
  0.50,
  0.90,
  0.975
)


# -----------------------------------------------------------------------------
# 7. CONSTRUCT ESTIMATION SAMPLE
# -----------------------------------------------------------------------------

# IMPORTANT:
#
# The January 2024 observation is excluded.
#
# Therefore the model only sees information available at the
# December 2023 forecast origin.

train <- dat[
  dat$TIME_PERIOD < forecast_date,
  ,
  drop = FALSE
]

y_train <- train$USA

forecast_origin <- max(
  train$TIME_PERIOD
)

cat("\n============================================================\n")
cat("ONE-STEP-AHEAD GAS FORECAST\n")
cat("============================================================\n")

cat(
  "Forecast target:   ",
  format(forecast_date, "%Y-%m-%d"),
  "\n",
  sep = ""
)

cat(
  "Forecast origin:   ",
  format(forecast_origin, "%Y-%m-%d"),
  "\n",
  sep = ""
)

cat(
  "Training observations: ",
  length(y_train),
  "\n",
  sep = ""
)


# -----------------------------------------------------------------------------
# 8. ESTIMATE CANDIDATE MODELS
# -----------------------------------------------------------------------------

# 2 distributions x 4 p values x 4 q values = 32 models.

candidate_results <- list()

for (distr_i in candidate_distributions) {
  
  for (p_i in candidate_p) {
    
    for (q_i in candidate_q) {
      
      cat(
        "Estimating ",
        distr_i,
        " GAS(",
        p_i,
        ",",
        q_i,
        ") ... ",
        sep = ""
      )
      
      fitted <- tryCatch(
        
        gas(
          y = y_train,
          distr = distr_i,
          param = candidate_param,
          scaling = scaling,
          regress = regress,
          p = p_i,
          q = q_i
        ),
        
        error = function(e) {
          NULL
        }
      )
      
      
      # -----------------------------------------------------------------------
      # SUCCESSFUL MODEL
      # -----------------------------------------------------------------------
      
      if (!is.null(fitted)) {
        
        loglik_i <- as.numeric(
          fitted$fit$loglik_sum
        )
        
        aic_i <- as.numeric(
          fitted$fit$aic
        )
        
        bic_i <- as.numeric(
          fitted$fit$bic
        )
        
        candidate_results[[length(candidate_results) + 1L]] <- list(
          
          model = fitted,
          
          distribution = distr_i,
          
          p = p_i,
          
          q = q_i,
          
          loglik = loglik_i,
          
          AIC = aic_i,
          
          BIC = bic_i,
          
          status = "success"
        )
        
        cat(
          "success; BIC = ",
          round(bic_i, 3),
          "\n",
          sep = ""
        )
        
      } else {
        
        # Failed models are retained in the selection table.
        
        candidate_results[[length(candidate_results) + 1L]] <- list(
          
          model = NULL,
          
          distribution = distr_i,
          
          p = p_i,
          
          q = q_i,
          
          loglik = NA_real_,
          
          AIC = NA_real_,
          
          BIC = NA_real_,
          
          status = "failed"
        )
        
        cat("FAILED\n")
      }
    }
  }
}


# -----------------------------------------------------------------------------
# 9. MODEL SELECTION TABLE
# -----------------------------------------------------------------------------

selection_results <- do.call(
  rbind,
  lapply(
    candidate_results,
    function(z) {
      
      data.frame(
        
        forecast_date = forecast_date,
        
        forecast_origin = forecast_origin,
        
        n_training_obs = length(y_train),
        
        distribution = z$distribution,
        
        p = z$p,
        
        q = z$q,
        
        loglik = z$loglik,
        
        AIC = z$AIC,
        
        BIC = z$BIC,
        
        status = z$status,
        
        stringsAsFactors = FALSE
      )
    }
  )
)
# -----------------------------------------------------------------------------
# 10. SELECT MODEL WITH LOWEST BIC
# -----------------------------------------------------------------------------

successful <- which(
  selection_results$status == "success" &
    is.finite(selection_results$BIC)
)

if (length(successful) == 0) {
  stop(
    "No candidate GAS model was successfully estimated."
  )
}

selected_row <- successful[
  which.min(selection_results$BIC[successful])
]

selected_model <- candidate_results[[selected_row]]$model

selected_distribution <- candidate_results[[selected_row]]$distribution
selected_p <- candidate_results[[selected_row]]$p
selected_q <- candidate_results[[selected_row]]$q
selected_loglik <- candidate_results[[selected_row]]$loglik
selected_aic <- candidate_results[[selected_row]]$AIC
selected_bic <- candidate_results[[selected_row]]$BIC


# -----------------------------------------------------------------------------
# 11. REPORT SELECTED MODEL
# -----------------------------------------------------------------------------

cat("\n============================================================\n")
cat("SELECTED MODEL\n")
cat("============================================================\n")

cat(
  "Distribution: ",
  selected_distribution,
  "\n",
  sep = ""
)

cat(
  "GAS order:    (",
  selected_p,
  ", ",
  selected_q,
  ")\n",
  sep = ""
)

cat(
  "Log-likelihood: ",
  round(selected_loglik, 4),
  "\n",
  sep = ""
)

cat(
  "AIC:            ",
  round(selected_aic, 4),
  "\n",
  sep = ""
)

cat(
  "BIC:            ",
  round(selected_bic, 4),
  "\n",
  sep = ""
)


# -----------------------------------------------------------------------------
# 12. ONE-STEP-AHEAD PROBABILISTIC FORECAST
# -----------------------------------------------------------------------------

forecast_object <- tryCatch(
  
  gas_forecast(
    gas_object = selected_model,
    method = "simulated_paths",
    t_ahead = 1L,
    rep_ahead = n_sim,
    quant = forecast_quantiles
  ),
  
  error = function(e) {
    
    cat(
      "\nForecast error:\n",
      e$message,
      "\n"
    )
    
    NULL
  }
)

if (is.null(forecast_object)) {
  
  stop(
    "The probabilistic forecast failed."
  )
}
# -----------------------------------------------------------------------------
# 13. EXTRACT PREDICTIVE DISTRIBUTION
# -----------------------------------------------------------------------------

cat("\nStructure of predictive forecast:\n")
str(forecast_object$forecast)

cat("\nDimensions of y_ahead_quant:\n")
print(dim(forecast_object$forecast$y_ahead_quant))

cat("\ny_ahead_quant:\n")
print(forecast_object$forecast$y_ahead_quant)

# -----------------------------------------------------------------------------
# 14. ACTUAL JANUARY 2024 VALUE
# -----------------------------------------------------------------------------

# This is retrieved only for reporting.
#
# It was NOT available to the model when the forecast was generated.

actual <- dat$USA[
  match(
    forecast_date,
    dat$TIME_PERIOD
  )
]

# -----------------------------------------------------------------------------
# 15. EXTRACT FORECAST MEAN AND SD
# -----------------------------------------------------------------------------

forecast_mean <- as.numeric(
  forecast_object$forecast$y_ahead_mean
)[1]

forecast_sd <- as.numeric(
  forecast_object$forecast$y_ahead_sd
)[1]

if (!is.finite(forecast_mean)) {
  stop("Could not extract a valid forecast mean.")
}

if (!is.finite(forecast_sd)) {
  stop("Could not extract a valid forecast standard deviation.")
}


# -----------------------------------------------------------------------------
# 16. EXTRACT FORECAST QUANTILES
# -----------------------------------------------------------------------------

quant_object <- forecast_object$forecast$y_ahead_quant

if (is.null(quant_object)) {
  stop("gas_forecast() did not return y_ahead_quant.")
}

cat("\nStructure of y_ahead_quant:\n")
str(quant_object)

cat("\nDimensions of y_ahead_quant:\n")
print(dim(quant_object))


# -----------------------------------------------------------------------------
# 17. CONVERT QUANTILES TO VECTOR
# -----------------------------------------------------------------------------

quant_dims <- dim(quant_object)

if (is.null(quant_dims)) {
  
  # Vector
  forecast_quantile_values <- as.numeric(
    quant_object
  )
  
} else if (length(quant_dims) == 2) {
  
  # Matrix
  
  if (quant_dims[1] == length(forecast_quantiles)) {
    
    forecast_quantile_values <- as.numeric(
      quant_object[, 1]
    )
    
  } else if (quant_dims[2] == length(forecast_quantiles)) {
    
    forecast_quantile_values <- as.numeric(
      quant_object[1, ]
    )
    
  } else {
    
    stop(
      paste(
        "Cannot identify quantile dimension.",
        "Dimensions:",
        paste(quant_dims, collapse = " x ")
      )
    )
  }
  
} else if (length(quant_dims) == 3) {
  
  # Array:
  # horizon x series x quantile
  
  forecast_quantile_values <- as.numeric(
    quant_object[1, 1, ]
  )
  
} else {
  
  stop(
    paste(
      "Unexpected number of dimensions:",
      length(quant_dims)
    )
  )
}


# -----------------------------------------------------------------------------
# 18. CHECK QUANTILES
# -----------------------------------------------------------------------------

if (
  length(forecast_quantile_values) !=
  length(forecast_quantiles)
) {
  
  stop(
    paste(
      "Expected",
      length(forecast_quantiles),
      "quantiles but extracted",
      length(forecast_quantile_values)
    )
  )
}


# -----------------------------------------------------------------------------
# 19. ASSIGN INDIVIDUAL QUANTILES
# -----------------------------------------------------------------------------

q025 <- forecast_quantile_values[1]

q10 <- forecast_quantile_values[2]

q50 <- forecast_quantile_values[3]

q90 <- forecast_quantile_values[4]

q975 <- forecast_quantile_values[5]


# -----------------------------------------------------------------------------
# 20. PRINT EXTRACTED FORECAST
# -----------------------------------------------------------------------------

cat("\n============================================================\n")
cat("EXTRACTED FORECAST\n")
cat("============================================================\n")

cat(
  "Mean:   ",
  round(forecast_mean, 4),
  "\n",
  sep = ""
)

cat(
  "SD:     ",
  round(forecast_sd, 4),
  "\n",
  sep = ""
)

cat(
  "Q2.5%:  ",
  round(q025, 4),
  "\n",
  sep = ""
)

cat(
  "Q10%:   ",
  round(q10, 4),
  "\n",
  sep = ""
)

cat(
  "Q50%:   ",
  round(q50, 4),
  "\n",
  sep = ""
)

cat(
  "Q90%:   ",
  round(q90, 4),
  "\n",
  sep = ""
)

cat(
  "Q97.5%: ",
  round(q975, 4),
  "\n",
  sep = ""
)