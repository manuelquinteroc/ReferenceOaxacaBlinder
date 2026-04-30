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
library(lightgbm)
library(fastDummies)

nl = new.env()
source(here('census', 'hlp_nonlinear.R'), local = nl)

# Load Data ---------------------------------------------------------------
icu = read_fst(here('census', 'temp', 'icu_clean.fst'))

fits_obs = read_fst(here('census', 'temp', 'icu_gbt_fits.fst')) |> as_tibble()


# Configure ---------------------------------------------------------------
flips = fits_obs |> 
  filter(sign(explained_0 * explained_1) != 1 | 
           sign(unexplained_0 * unexplained_1) != 1)

design = flips |> 
  select(subset_name, 
         subset_value,
         pop_name,
         y_name,
         algo_name)

already_fit = length(list.files(here('census', 'temp', 'icu_gbt_boots')))

x_names_train = c("Age", "ICUType", "HR", "NIMAP", "Temp", "Urine")

icu_train = icu |> 
  select(all_of(unique(design$subset_name)),
         all_of(unique(design$y_name)),
         all_of(unique(design$pop_name)),
         all_of(x_names_train))

# Fit ---------------------------------------------------------------------
message("Fitting...")
plan(multicore, workers = min(5, future::availableCores()))
for (ii in seq(already_fit + 1, nrow(design))) {
  fit = mutate(design[ii,],
               fit_col = pmap(design[ii,], 
                              nl$twoway_subset,
                              x_names  = x_names_train,
                              data     = icu_train,
                              n_boot   = 1000,
                              verbose_outer = T,
                              parallel_inner = T),
               fit_col = map(fit_col, 'boot'))
  
  fit = unnest(fit, fit_col)
  
  file = here('census', 'temp', 'icu_gbt_boots', glue('{ii}.fst'))
  write_fst(fit, file)
  
  message("Wrote: ", file)
}

# Done --------------------------------------------------------------------
message("Done.")
tictoc::toc()
