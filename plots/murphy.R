murphy_scores <- function(forecasts, alphas = c(0.1, 0.5, 0.9)) {
  bind_rows(lapply(alphas, function(alpha) {
    quantile_col <- paste0("q", alpha)
    thresholds <- seq(
      min(c(forecasts$actual, forecasts[[quantile_col]]), na.rm = TRUE),
      max(c(forecasts$actual, forecasts[[quantile_col]]), na.rm = TRUE),
      length.out = 101
    )

    bind_rows(lapply(thresholds, function(theta) {
      bind_rows(lapply(split(forecasts, forecasts$model), function(model_forecasts) {
        outcome_below <- model_forecasts$actual <= theta
        forecast_below <- model_forecasts[[quantile_col]] <= theta

        tibble(
          model = model_forecasts$model[1],
          threshold = theta,
          score = mean(
            2 * (as.numeric(outcome_below) - alpha) *
              (as.numeric(outcome_below) - as.numeric(forecast_below))
          )
        )
      }))
    })) %>%
      mutate(alpha = alpha)
  }))
}

plot_murphy <- function(murphy_data, indicator, horizon, countries) {
  ggplot(murphy_data, aes(threshold, score, color = model)) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~alpha, nrow = 1, labeller = label_bquote(alpha == .(alpha))) +
    labs(
      x = "threshold", y = "mean elementary quantile score",
      title = paste0(
        "Murphy diagrams - ", indicator, " - horizon ", horizon,
        " - ", paste(countries, collapse = ", ")
      ),
      color = "model"
    ) +
    plot_theme()
}
