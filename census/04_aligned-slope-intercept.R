# -------------------------------------------------------------------------
# Check the aligned slope-intercept condition
# -------------------------------------------------------------------------


# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)


# Helpers -----------------------------------------------------------------
# Assumption 5.1 ("aligned slope-intercept"): the intercept gap (delta_alpha)
# points in the same direction as both reference groups' explained components.
# When this holds, the sign of the explained part is stable across reference groups.
is_aligned = function(delta_alpha, explained_0, explained_1) {

  (explained_0 * explained_1 > 0) &
    (sign(delta_alpha) == sign(explained_0)) &
    (sign(delta_alpha) == sign(explained_1))

}


# Read Data ---------------------------------------------------------------
fits  = read_fst(here('census', 'temp', 'nonlinear_fits.fst')) |> as_tibble()
align = read_fst(here('census', 'temp', 'ols_aligned.fst')) |> as_tibble()

# Same size/magnitude filter as 02 for a comparable base of cases
fits_align = fits |>
  filter(n_0 > 50, n_1 > 50,
         abs(delta_y) > 0.01,
         algo_name == 'ols') |> 
  inner_join(align)

# Summary stats quoted in Section 5.4 of the paper
align_tb = fits_align |>
  group_by(y_name) |> 
  summarize(pct_aligned = mean(aligned),
            n_aligned   = sum(aligned),
            n_total     = n())

# save
write_csv(align_tb, here('census', 'out', 'aligned_stats.csv'))