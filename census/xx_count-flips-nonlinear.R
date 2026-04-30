# -------------------------------------------------------------------------
# Hypothesis testing based on bootstrap
# -------------------------------------------------------------------------

# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(scales)
library(forcats)
library(ggplot2)

# Read Data ---------------------------------------------------------------
fits  = read_fst(here('census', 'temp', 'nonlinear_fits.fst')) |> as_tibble()

boots = read_fst(here('census', 'temp', 'nonlinear_boots.fst')) |> as_tibble()


# Standard Errors ---------------------------------------------------------
boot_se = boots |> 
  pivot_longer(cols = c(starts_with('delta'),
                        ends_with('_0'),
                        ends_with('_1')),
               names_to  = 'stat_name',
               values_to = 'stat_value_boot') |> 
  group_by(subset_name, subset_value, pop_name, algo_name, y_name, 
           stat_name) |> 
  summarize(stat_se = sd(stat_value_boot),
            .groups = 'drop')

# Append SEs  -------------------------------------------------------------
flips = fits |> 
  filter(n_0 > 50, n_1 > 50, abs(delta_y) > 0.01) |> 
  mutate(is_flip_explained   = explained_1 * explained_0 < 0,
         is_flip_unexplained = unexplained_0 * unexplained_1 < 0,
         is_flip = is_flip_explained | is_flip_unexplained)

flip_counts = flips |> 
  group_by(y_name, algo_name) |> 
  summarize(n_fits = n(),
            n_flip             = sum(is_flip),
            n_flip_explained   = sum(is_flip_explained),
            n_flip_unexplained = sum(is_flip_unexplained),
            .groups = 'drop')

write_csv(flip_counts, here('census', 'out', 'flip_counts.csv'))
