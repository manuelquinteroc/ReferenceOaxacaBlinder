# -------------------------------------------------------------------------
# Fit Oaxaca-Blinder decompositions
# -------------------------------------------------------------------------

# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(fixest)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(furrr)

ob = new.env()
source(here('census', 'hlp_oaxaca-blinder.R'), local = ob)

# Load Data ---------------------------------------------------------------
acs = read_fst(here('census', 'temp', 'acs16_workforce.fst')) |> as_tibble()

# Configure ---------------------------------------------------------------
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
                  x_name     = c('educ_bach', 'mar'),
                  y_name     = c('pincp', 'hicov'))


# Fit ---------------------------------------------------------------------
message("Fitting...")
plan(multicore, workers = min(4, future::availableCores()))
# n_boot = 0: skip bootstrapping here; only sign-flip cases get bootstrapped in 02
fits = mutate(design,
                 fit = future_pmap(design, ob$twoway_subset,
                                   data   = acs,
                                   n_boot = 0))

fits_obs = fits |>
  mutate(fit = map(fit, "obs")) |>
  unnest(fit) |>
  # Pivot so ref_value=0 and ref_value=1 become separate columns;
  # sign flips are then detectable by comparing explained_0 vs explained_1 in one row
  pivot_wider(names_from  = c('ref_value'),
              values_from = c('explained', 'unexplained', 'n', 'r2'))


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(fits_obs, here('census', 'temp', 'ob_fits.fst'))

# Done --------------------------------------------------------------------
message("Done.")
