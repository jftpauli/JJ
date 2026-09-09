plot_wis_decomposition <- function(scores_summary, indicator, horizon, countries) {
  decomp <- scores_summary %>%
    select(model, sharpness, calibration) %>%
    pivot_longer(-model, names_to = "component", values_to = "value")

  ggplot(decomp, aes(model, value, fill = component)) +
    geom_col() +
    labs(
      y = "WIS (pooled across countries)", x = NULL,
      title = paste0(
        "WIS decomposition - ", indicator, " - horizon ", horizon,
        " - ", paste(countries, collapse = ", ")
      )
    ) +
    plot_theme()
}
