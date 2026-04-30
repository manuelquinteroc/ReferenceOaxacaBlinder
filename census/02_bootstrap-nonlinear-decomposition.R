# -------------------------------------------------------------------------
# Bootstrap resampling for nonlinear analyses with sign flips where
# groups have n > 50 observations and group differences are large enough
# -------------------------------------------------------------------------
tictoc::tic()

# Libraries ---------------------------------------------------------------
library(here)
library(glue)
library(fst)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(furrr)
library(lightgbm)
library(fastDummies)

nl = new.env()
source(here('census', 'hlp_nonlinear.R'), local = nl)

# Load Data ---------------------------------------------------------------
acs_raw = read_fst(here('census', 'temp', 'acs16_workforce.fst')) |> as_tibble()
# Drop NAICS3; too many factor levels, which slows ML model fits
acs     = acs_raw |> select(-naics_3)

fits_obs = read_fst(here('census', 'temp', 'nonlinear_fits.fst')) |> as_tibble()


# Configure ---------------------------------------------------------------
# Same size/magnitude filter as 02 (linear version); applied to nonlinear fits
fits_sized = fits_obs |>
  filter(n_0 > 50, n_1 > 50, abs(delta_y) > 0.01)

# Same sign-flip criterion as 02
flips_sized = fits_sized |>
  filter(sign(explained_0 * explained_1) != 1 |
           sign(unexplained_0 * unexplained_1) != 1)

design = flips_sized |>
  select(subset_name,
         subset_value,
         pop_name,
         y_name,
         algo_name) |> 
  arrange(desc(algo_name == 'ols'),
          desc(algo_name == 'glm'))


already_fit = length(list.files(here('census', 'temp', 'nonlinear_boots')))


# Fit ---------------------------------------------------------------------
message("Fitting...")
plan(multicore, workers = min(6, future::availableCores()))
for (ii in seq(already_fit + 1, nrow(design))) {
  fit = mutate(design[ii,],
               fit_col = pmap(design[ii,], 
                              nl$twoway_subset, 
                              data = acs, n_boot = 1000, 
                              verbose_outer  = T, 
                              verbose_inner  = F, 
                              parallel_inner = T),
               fit_col = map(fit_col, 'boot'))
  
  fit = unnest(fit, fit_col)
  
  file = here('census', 'temp', 'nonlinear_boots', glue('{ii}.fst'))
  write_fst(fit, file)
  
  message("Wrote: ", file)
}


# Combine -----------------------------------------------------------------
boots = list.files(here('census', 'temp', 'nonlinear_boots'), full.names = TRUE) |> 
  map(read_fst) |> 
  list_rbind()


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(boots, here('census', 'temp', 'nonlinear_boots.fst'))


# Done --------------------------------------------------------------------
message("Done.")
tictoc::toc()
