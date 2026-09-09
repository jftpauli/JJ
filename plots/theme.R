plot_theme <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(
        face = "bold", size = rel(1.05), margin = margin(b = 10)
      ),
      plot.subtitle = element_text(color = "grey35"),
      axis.title = element_text(face = "bold", color = "grey20"),
      axis.text = element_text(color = "grey25"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey85", linewidth = 0.3),
      strip.text = element_text(face = "bold", color = "grey20"),
      strip.background = element_rect(fill = "grey93", color = NA),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      panel.spacing = unit(0.6, "lines")
    )
}
