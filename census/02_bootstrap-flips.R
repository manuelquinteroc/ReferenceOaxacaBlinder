# -------------------------------------------------------------------------
# Bootstrap resampling for Oaxaca-Blinder analyses with sign flips where
# groups have n > 50 observations and group differences are large enough
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
library(furrr)

ob = new.env()
source(here('census', 'hlp_oaxaca-blinder.R'), local = ob)

# Load Data ---------------------------------------------------------------
acs = read_fst(here('census', 'temp', 'acs16_workforce.fst')) |> as_tibble()

fits_obs = read_fst(here('census', 'temp', 'ob_fits.fst')) |> as_tibble()


# Configure ---------------------------------------------------------------
fits_sized = fits_obs |> 
  filter(!is.na(delta_beta),
         n_0 > 50, n_1 > 50,
         abs(delta_y) > 0.01,
         abs(explained_0) > 0.01,
         abs(unexplained_0) > 0.01,
         abs(explained_1) > 0.01,
         abs(unexplained_1) > 0.01)

flips_sized = fits_sized |> 
  filter(sign(explained_0 * explained_1) != 1 | 
           sign(unexplained_0 * unexplained_1) != 1)

design = flips_sized |> 
  select(subset_name, 
         subset_value,
         pop_name,
         x_name,
         y_name)


# Fit ---------------------------------------------------------------------
message("Fitting...")
plan(multicore, workers = min(10, future::availableCores()))
fits = mutate(design,
              fit = pmap(design,
                         ob$twoway_subset,
                         data   = acs,
                         n_boot = 20000))

boots = fits |> 
  mutate(fit = map(fit, "boot")) |> 
  unnest(fit) |> 
  pivot_wider(names_from  = c('ref_value'),
              values_from = c('explained', 'unexplained', 'n', 'r2'))


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(boots, here('census', 'temp', 'ob_boots.fst'))

# Done --------------------------------------------------------------------
message("Done.")
