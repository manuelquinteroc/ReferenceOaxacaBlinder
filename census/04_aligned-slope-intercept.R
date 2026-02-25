# -------------------------------------------------------------------------
# Check the aligned slope-intercept condition
# -------------------------------------------------------------------------


# Libraries ---------------------------------------------------------------
renv::load()
library(here)
library(fst)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)


# Helpers -----------------------------------------------------------------
is_aligned = function(delta_alpha, explained_0, explained_1) {
  
  
  (explained_0 * explained_1 > 0) &
    (sign(delta_alpha) == sign(explained_0)) & 
    (sign(delta_alpha) == sign(explained_1))
  
}


# Read Data ---------------------------------------------------------------
fits  = read_fst(here('census', 'temp', 'ob_fits.fst')) |> as_tibble()

fits_sized = fits |> 
  filter(!is.na(delta_beta),
         n_0 > 50, n_1 > 50,
         abs(delta_y) > 0.01,
         abs(explained_0) > 0.01,
         abs(unexplained_0) > 0.01,
         abs(explained_1) > 0.01,
         abs(unexplained_1) > 0.01)

# summary stats for text
align_tb = fits_sized |> 
  mutate(aligned = is_aligned(delta_alpha, explained_0, explained_1)) |> 
  summarize(pct_aligned = mean(aligned), 
            n_aligned   = sum(aligned), 
            n_total     = n())

# save
write_csv(align_tb, here('census', 'out', 'aligned_stats.csv'))