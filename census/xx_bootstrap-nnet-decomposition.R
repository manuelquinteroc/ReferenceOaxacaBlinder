# -------------------------------------------------------------------------
# Bootstrap resampling for nonlinear analyses with sign flips where
# groups have n > 50 observations and group differences are large enough
# -------------------------------------------------------------------------
tictoc::tic()

# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(readr)
library(glue)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(furrr)
library(brulee)
library(fastDummies)

nl = new.env()
source(here('census', 'hlp_nonlinear.R'), local = nl)

# Load Data ---------------------------------------------------------------
acs_raw = read_fst(here('census', 'temp', 'acs16_workforce.fst')) |> as_tibble()
acs     = acs_raw |> select(-naics_2, -naics_3, -race_bw)

fits_obs = read_fst(here('census', 'temp', 'nonlinear_fits_net.fst')) |> as_tibble()


# Configure ---------------------------------------------------------------
fits_sized = fits_obs |> 
  filter(n_0 > 50, n_1 > 50,
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
         y_name,
         algo_name)

already_fit = length(list.files(here('census', 'temp', 'neural_boots')))

# Fit ---------------------------------------------------------------------
message("Fitting...")
for (ii in seq(already_fit + 1, nrow(design))) {
  fit = mutate(design[ii,],
               fit_col = pmap(design[ii,], 
                              nl$twoway_subset, 
                              data = acs, n_boot = 1000, 
                              verbose_outer = T, verbose_inner = T, 
                              parallel_inner = F),
               fit_col = map(fit_col, 'boot'))
  
  fit = unnest(fit, fit_col)
  
  file = here('census', 'temp', 'neural_boots', glue('{ii}.fst'))
  write_fst(fit, file)
  
  message("Wrote: ", file)
}

# Done --------------------------------------------------------------------
message("Done.")
tictoc::toc()
