# ============================================================
# Packages
# ============================================================

# install.packages(c("dplyr", "gt", "webshot2"))

library(dplyr)
library(gt)

# ============================================================
# Raw data
# ============================================================

L0 <- list(
  PPO = list(
    success = c(0.01,0.00,0.08,0.20,0.06),
    violation = c(0.10,0.13,0.18,0.15,0.02),
    ep_len = c(1586,766,1779,1415,963)
  ),
  PPO_Lagrangian = list(
    success = c(0.00,0.23,0.05,0.22,0.24),
    violation = c(0.03,0.19,0.19,0.34,0.14),
    ep_len = c(662,837,1538,820,796)
  ),
  GenZ_LTL = list(
    success = c(0.44,0.46,0.45,0.44,0.45),
    violation = c(0.22,0.18,0.10,0.19,0.08),
    ep_len = c(774,712,922,737,837)
  )
)

L1 <- list(
  PPO = list(
    success = c(0.01,0.00,0.00,0.00,0.00),
    violation = c(0.03,0.01,0.111,0.05,0.28),
    ep_len = c(721,467,1869,2319,2275)
  ),
  PPO_Lagrangian = list(
    success = c(0.00,0.00,0.00,0.00,0.00),
    violation = c(0.25,0.13,0.17,0.07,0.24),
    ep_len = c(922,881,576,787,1114)
  )
)

# ============================================================
# Formatting helper
# ============================================================

paper_fmt <- function(x, digits = 2) {
  sprintf(
    "%.2f ± %.2f",
    mean(x),
    sd(x)
  )
}

paper_fmt_len <- function(x) {
  sprintf(
    "%.2f ± %.2f",
    mean(x),
    sd(x)
  )
}

# ============================================================
# Summary table
# ============================================================

make_row <- function(level, d) {
  
  tibble(
    Level = level,
    
    PPO_s   = paper_fmt(d$PPO$success),
    PPO_v   = paper_fmt(d$PPO$violation),
    PPO_mu  = paper_fmt_len(d$PPO$ep_len),
    
    PPOL_s  = paper_fmt(d$PPO_Lagrangian$success),
    PPOL_v  = paper_fmt(d$PPO_Lagrangian$violation),
    PPOL_mu = paper_fmt_len(d$PPO_Lagrangian$ep_len),
    
    GENZ_s = if ("GenZ_LTL" %in% names(d))
      paper_fmt(d$GenZ_LTL$success) else NA_character_,
    
    GENZ_v = if ("GenZ_LTL" %in% names(d))
      paper_fmt(d$GenZ_LTL$violation) else NA_character_,
    
    GENZ_mu = if ("GenZ_LTL" %in% names(d))
      paper_fmt_len(d$GenZ_LTL$ep_len) else NA_character_
  )
}

tbl <- bind_rows(
  make_row("Level 0", L0),
  make_row("Level 1", L1)
)

bold_row_best <- function(row, cols, direction = "max") {
  
  vals <- sapply(
    row[cols],
    function(x) as.numeric(trimws(sub(" ±.*", "", x)))
  )
  
  best <- if (direction == "max") {
    max(vals, na.rm = TRUE)
  } else {
    min(vals, na.rm = TRUE)
  }
  
  for (i in seq_along(cols)) {
    if (!is.na(vals[i]) && vals[i] == best) {
      row[[cols[i]]] <- paste0(
        "<span style='font-weight:600'>",
        row[[cols[i]]],
        "</span>"
      )
    }
  }
  
  row
}

small_sd <- function(x) {
  
  x <- gsub(
    " ± ",
    "<span style='font-size:65%'> ± ",
    x
  )
  
  x <- gsub(
    "$",
    "</span>",
    x
  )
  
  x
}

tbl <- as.data.frame(tbl)

tbl <- t(apply(tbl, 1, function(x) {
  
  x <- as.list(x)
  
  x <- bold_row_best(
    x,
    c("PPO_s", "PPOL_s", "GENZ_s"),
    "max"
  )
  
  x <- bold_row_best(
    x,
    c("PPO_v", "PPOL_v", "GENZ_v"),
    "min"
  )
  
  x <- bold_row_best(
    x,
    c("PPO_mu", "PPOL_mu", "GENZ_mu"),
    "min"
  )
  
  unlist(x)
  
})) |> 
  as.data.frame(stringsAsFactors = FALSE)

names(tbl) <- c(
  "Level",
  "PPO_s", "PPO_v", "PPO_mu",
  "PPOL_s", "PPOL_v", "PPOL_mu",
  "GENZ_s", "GENZ_v", "GENZ_mu"
)

tbl <- tbl |>
  mutate(
    across(
      -Level,
      small_sd
    )
  )

# ============================================================
# Table
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
    
    heading.border.bottom.width = px(1),
    
    table.border.top.width = px(2),
    table.border.bottom.width = px(2),
    
    table_body.hlines.width = px(1),
    
    column_labels.border.top.width = px(1),
    column_labels.border.bottom.width = px(1),
    
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

