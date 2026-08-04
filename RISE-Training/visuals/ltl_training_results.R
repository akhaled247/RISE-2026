# ============================================================
# Packages
# ============================================================

library(dplyr)
library(gt)

# ============================================================
# Raw data (revised from experiment logs)
# ============================================================

L0 <- list(
  
  PPO = list(
    success = c(0.050, 0.000, 0.020, 0.030, 0.250),
    violation = c(0.160, 0.640, 0.710, 0.100, 0.090),
    ep_len = c(1028.8, NA, 1426.0, 1107.3, 1069.2)
  ),
  
  PPO_Lagrangian = list(
    success = c(0.140, 0.280, 0.140, 0.330, 0.470),
    violation = c(0.190, 0.130, 0.070, 0.210, 0.220),
    ep_len = c(1005.4, 941.5, 810.4, 879.1, 892.5)
  ),
  
  GenZ_LTL = list(
    success = c(0.730, 0.850, 0.780, 0.750, 0.780),
    violation = c(0.160, 0.050, 0.040, 0.160, 0.020),
    ep_len = c(776.000, 766.718, 890.731, 765.800, 826.974)
  )
)


L1 <- list(
  
  PPO = list(
    success = c(0.010, 0.010, 0.000, 0.000, 0.000),
    violation = c(0.690, 0.740, 0.170, 0.190, 0.140),
    ep_len = c(637.0, 625.0, NA, NA, NA)
  ),
  
  PPO_Lagrangian = list(
    success = c(0.000, 0.010, 0.030, 0.000, 0.000),
    violation = c(0.570, 0.600, 0.570, 0.870, 0.510),
    ep_len = c(NA, 2259.0, 1169.7, NA, NA)
  ),
  
  GenZ_LTL = list(
  success = c(0.020, 0.050, 0.100, 0.010, 0.030),
  violation = c(0.910, 0.910, 0.870, 0.950, 0.900),
  ep_len = c(542.500, 652.600, 620.100, 716.000, 664.667)
  )
)


# ============================================================
# Formatting
# ============================================================

paper_fmt <- function(x) {
  
  sprintf(
    "%.2f ± %.2f",
    mean(x, na.rm = TRUE),
    sd(x, na.rm = TRUE)
  )
  
}


paper_fmt_len <- function(x) {
  
  sprintf(
    "%.2f ± %.2f",
    mean(x, na.rm = TRUE),
    sd(x, na.rm = TRUE)
  )
  
}


# ============================================================
# Create summary table
# ============================================================

make_row <- function(level, d) {
  
  tibble(
    Level = level,
    
    PPO_s = paper_fmt(d$PPO$success),
    PPO_v = paper_fmt(d$PPO$violation),
    PPO_mu = paper_fmt(d$PPO$ep_len),
    
    PPOL_s = paper_fmt(d$PPO_Lagrangian$success),
    PPOL_v = paper_fmt(d$PPO_Lagrangian$violation),
    PPOL_mu = paper_fmt(d$PPO_Lagrangian$ep_len),
    
    GENZ_s = paper_fmt(d$GenZ_LTL$success),
    GENZ_v = paper_fmt(d$GenZ_LTL$violation),
    GENZ_mu = paper_fmt(d$GenZ_LTL$ep_len)
  )
}


tbl <- bind_rows(
  make_row("Level 0", L0),
  make_row("Level 1", L1)
)


# ============================================================
# Bold best values per level
# ============================================================

bold_row_best <- function(row, cols, direction = "max") {
  
  values <- sapply(
    cols,
    function(col) {
      as.numeric(sub(" ±.*", "", row[[col]]))
    }
  )
  
  best <- if (direction == "max") {
    max(values, na.rm = TRUE)
  } else {
    min(values, na.rm = TRUE)
  }
  
  
  for (col in cols) {
    
    value <- as.numeric(
      sub(" ±.*", "", row[[col]])
    )
    
    if (value == best) {
      
      row[[col]] <- paste0(
        "<span style='font-weight:600'>",
        row[[col]],
        "</span>"
      )
      
    }
  }
  
  row
}


tbl <- split(tbl, seq_len(nrow(tbl))) |>
  lapply(function(row) {
    
    row <- bold_row_best(
      row,
      c("PPO_s", "PPOL_s", "GENZ_s"),
      "max"
    )
    
    row <- bold_row_best(
      row,
      c("PPO_v", "PPOL_v", "GENZ_v"),
      "min"
    )
    
    row <- bold_row_best(
      row,
      c("PPO_mu", "PPOL_mu", "GENZ_mu"),
      "min"
    )
    
    row
    
  }) |>
  bind_rows()


# ============================================================
# Make standard deviation smaller
# ============================================================

small_sd <- function(x) {
  
  sub(
    " ± ",
    "<span style='font-size:65%'> ± ",
    x
  ) |>
    paste0("</span>")
  
}


tbl <- tbl |>
  mutate(
    across(
      -Level,
      small_sd
    )
  )


# ============================================================
# GT table
# ============================================================

gt_tbl <-
  gt(tbl) |>
  
  fmt_markdown(
    columns = everything()
  ) |>
  
  cols_label(
    
    Level = "",
    
    PPO_s = md("&eta;<sub>s</sub> &uarr;"),
    PPO_v = md("&eta;<sub>v</sub> &darr;"),
    PPO_mu = md("&mu; &darr;"),
    
    PPOL_s = md("&eta;<sub>s</sub> &uarr;"),
    PPOL_v = md("&eta;<sub>v</sub> &darr;"),
    PPOL_mu = md("&mu; &darr;"),
    
    GENZ_s = md("&eta;<sub>s</sub> &uarr;"),
    GENZ_v = md("&eta;<sub>v</sub> &darr;"),
    GENZ_mu = md("&mu; &darr;")
    
  ) |>
  
  tab_spanner(
    label = md("**PPO**"),
    columns = PPO_s:PPO_mu
  ) |>
  
  tab_spanner(
    label = md("**PPO-Lagrangian**"),
    columns = PPOL_s:PPOL_mu
  ) |>
  
  tab_spanner(
    label = md("**GenZ-LTL**"),
    columns = GENZ_s:GENZ_mu
  ) |>
  
  opt_table_font(
    font = c(
      "Times New Roman",
      "Times",
      "serif"
    )
  ) |>
  
  tab_options(
    
    table.font.size = px(22),
    
    table.border.top.width = px(2),
    table.border.bottom.width = px(2),
    
    heading.border.bottom.width = px(1),
    
    column_labels.border.top.width = px(1),
    column_labels.border.bottom.width = px(1),
    
    table_body.hlines.width = px(1),
    
    table_body.vlines.width = px(0),
    column_labels.vlines.width = px(0),
    
    data_row.padding = px(12)
  )


# ============================================================
# Export
# ============================================================

gtsave(
  gt_tbl,
  "poster_table.png",
  zoom = 4,
  vwidth = 2400,
  vheight = 700
)


gt_tbl