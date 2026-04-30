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
library(furrr)
library(lightgbm)
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
plan(multicore, workers = 8)
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
                  y_name     = c('pincp', 'hicov')) |> 
  # only fit glm (logistic) for binary outcomes; for continuous outcomes glm = ols, already fit
  mutate(algo_name = map(y_name, \(yn) switch(yn, 
                                              pincp = c('ols', 'gbt'), 
                                              hicov = c('ols', 'glm', 'gbt')))) |> 
  unnest(algo_name)

# n_boot = 0: bootstrapping is deferred to 06 for sign-flip cases only
fits = design |>
  mutate(fit = future_pmap(design,
                           nl$twoway_subset,
                           data     = acs,
                           n_boot   = 0,
                           .progress = TRUE,
                           .options = furrr_options(seed = 1)))

fits_obs = fits |> 
  mutate(fit = map(fit, "obs")) |> 
  unnest(fit)


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(fits_obs, here('census', 'temp', 'nonlinear_fits.fst'))


# Done --------------------------------------------------------------------
message("Done.")
tictoc::toc()