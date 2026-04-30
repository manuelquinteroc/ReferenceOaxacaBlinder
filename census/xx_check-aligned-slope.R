# -------------------------------------------------------------------------
# Nonlinear models for decomposing a difference in outcomes
# -------------------------------------------------------------------------
tictoc::tic()


# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(fastDummies)


# Helpers -----------------------------------------------------------------
nl = new.env()
source(here('census', 'hlp_nonlinear.R'), local = nl)


# Load Data ---------------------------------------------------------------
acs_raw = read_fst(here('census', 'temp', 'acs16_workforce.fst')) |> as_tibble()
# Drop NAICS3; too many factor levels, which slows ML model fits
acs     = acs_raw |> select(-naics_3)


# Fit Decompositions ------------------------------------------------------
message("Fitting decompositions...")
subsets = acs |>
  select(st, naics_2) |>
  pivot_longer(everything(),
               names_to = 'subset_name',
               values_to = 'subset_value',
               values_transform = as.character) |>
  distinct() |>
  arrange(subset_name, subset_value)

# Full Cartesian product: every combination of geographic/industry subset,
# population variable, covariate, and outcome to decompose
design = crossing(subsets,
                  pop_name   = c('sex_female', 'race_bw', 'immigrant'),
                  y_name     = c('pincp', 'hicov'))

# n_boot = 0: bootstrapping is deferred to 06 for sign-flip cases only
fits = design |>
  mutate(aligned = pmap_lgl(design,
                            nl$aligned_slope_intercept,
                            data     = acs))


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(fits, here('census', 'temp', 'ols_aligned.fst'))


# Done --------------------------------------------------------------------
message("Done.")
tictoc::toc()