plot_model_comparison <- function(forecasts, historical, indicator, horizon, plot_country) {
  plot_data <- forecasts %>%
    filter(country == plot_country) %>%
    select(model, target_date, q0.05, q0.95, q0.5, actual) %>%
    rename(lower = q0.05, upper = q0.95, median = q0.5)

  zoom_start <- min(plot_data$target_date) - 365

  ggplot() +
    geom_line(
      data = filter(historical, date >= zoom_start),
      aes(x = date, y = value), color = "grey40", linewidth = 0.4
    ) +
    geom_ribbon(
      data = plot_data, aes(x = target_date, ymin = lower, ymax = upper),
      fill = "#2a78d6", alpha = 0.25
    ) +
    geom_line(
      data = plot_data, aes(x = target_date, y = median),
      color = "#2a78d6", linewidth = 0.6
    ) +
    geom_point(
      data = plot_data, aes(x = target_date, y = actual),
      color = "black", size = 0.7
    ) +
    facet_wrap(~model, ncol = 1, strip.position = "right") +
    labs(
      x = NULL,
      y = paste(toupper(indicator), "-", plot_country),
      title = paste0(
        "Model comparison (", plot_country, " only, for intuition) - ",
        indicator, " - horizon ", horizon
      )
    ) +
    plot_theme()
}
