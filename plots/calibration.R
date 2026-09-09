plot_pit <- function(forecasts, indicator, horizon, countries) {
  ggplot(forecasts, aes(pit)) +
    geom_histogram(
      aes(y = after_stat(density)), breaks = seq(0, 1, by = 0.1),
      fill = "#2a78d6", color = "white"
    ) +
    geom_hline(yintercept = 1, linetype = 2, color = "grey40") +
    facet_wrap(~model) +
    labs(
      x = "PIT", y = "density",
      title = paste0(
        "PIT histograms - ", indicator, " - horizon ", horizon,
        " - ", paste(countries, collapse = ", ")
      )
    ) +
    plot_theme()
}

plot_coverage <- function(coverage, indicator, horizon, countries) {
  ggplot(coverage, aes(nominal_coverage, empirical_coverage, color = model)) +
    geom_abline(linetype = 2, color = "grey50") +
    geom_line() +
    geom_point(size = 1.5) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      x = "nominal interval coverage", y = "empirical coverage",
      title = paste0(
        "Interval coverage vs nominal - ", indicator, " - horizon ", horizon,
        " - ", paste(countries, collapse = ", ")
      )
    ) +
    plot_theme()
}
