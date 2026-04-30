# -------------------------------------------------------------------------
# Nonlinear models for decomposing a difference in outcomes
# -------------------------------------------------------------------------
tictoc::tic()


# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(readr)
library(stringr)
library(glue)
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
icu = read_fst(here('census', 'temp', 'icu_clean.fst'))

subsets = icu |>   
  select(starts_with('subset'),
         starts_with('rand'),
         ICUType, age_bin, hr_bin, map_bin,
         temp_bin, urine_bin, saps_bin) |> 
  pivot_longer(everything(), 
               names_to = 'subset_name', 
               values_to = 'subset_value', 
               values_transform = as.character) |> 
  distinct() |> 
  filter(!is.na(subset_value)) |> 
  arrange(subset_name, subset_value)

design = crossing(subsets,
                  pop_name  = c('Gender'),
                  y_name    = c('In-hospital_death'),
                  algo_name = c('ols', 'glm'))

# Fit Decompositions ------------------------------------------------------
message("Fitting decompositions...")
# plan(multicore, workers = 8)

x_names_train = c("Age", "ICUType", "HR", "NIMAP", "Temp", "Urine")

icu_train = icu |> 
  select(all_of(unique(design$subset_name)),
         all_of(unique(design$y_name)),
         all_of(unique(design$pop_name)),
         all_of(x_names_train))

fits = design |> 
  mutate(fit = future_pmap(design, 
                           nl$twoway_subset,
                           x_names  = x_names_train,
                           data     = icu_train,
                           n_boot   = 0,
                           .options = furrr_options(seed = 1)))
fits_obs = fits |> 
  mutate(fit = map(fit, "obs")) |> 
  unnest(fit)


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(fits_obs, here('census', 'temp', 'icu_gbt_fits.fst'))


# Done --------------------------------------------------------------------
message("Done.")
tictoc::toc()