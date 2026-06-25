install.packages("ggsci")

library(tidyverse)
library(ggsci)

df <- read.csv("data/run2_cell_counts.csv")

comparison_plot <- function(xvar, yvar, xlab, ylab) {

  r2 <- summary(lm(df[[yvar]] ~ df[[xvar]]))$r.squared

  ggplot(df, aes(x = .data[[xvar]], y = .data[[yvar]])) +
    geom_point(
      aes(color = condition, shape = time_period),
      size = 3,
      alpha = 0.85
    ) +
    scale_color_iterm("Vaughn") +
    geom_smooth(
      method = "lm",
      se = FALSE,
      color = "black",
      linewidth = 0.6
    ) +
    annotate(
      "text",
      x = min(df[[xvar]], na.rm = TRUE),
      y = max(df[[yvar]], na.rm = TRUE),
      label = paste0("R² = ", round(r2, 3)),
      hjust = 0,
      vjust = 1,
      size = 4
    ) +
    labs(x = xlab, y = ylab) +
    theme_classic() +
    theme(
      panel.border = element_rect(color = "black", fill = NA),
      legend.position = "right"
    )
}

p1 <- comparison_plot("manual", "i3t3",  "Manual count", "i3t3 count")
p2 <- comparison_plot("imagej", "i3t3",  "ImageJ count", "i3t3 count")
p3 <- comparison_plot("imagej", "manual", "ImageJ count", "Manual count")

p1
p2
p3

ggsave("outputs/A_i3t3_vs_manual.png",  p1, dpi = 1000)
ggsave("outputs/C_manual_vs_imagej.png", p3, dpi = 1000)
