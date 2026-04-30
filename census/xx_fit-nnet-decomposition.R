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
library(brulee)
library(fastDummies)


# Helpers -----------------------------------------------------------------
nl = new.env()
source(here('census', 'hlp_nonlinear.R'), local = nl)


# Load Data ---------------------------------------------------------------
acs_raw = read_fst(here('census', 'temp', 'acs16_workforce.fst')) |> as_tibble()
acs     = acs_raw |> select(-naics_2, -naics_3, -race_bw)


# Fit Decompositions ------------------------------------------------------
message("Fitting decompositions...")
config = crossing(subset_name    = "st",
                  subset_value   = unique(acs$st),
                  pop_name       = c("sex_female", "immigrant", "educ_bach"),
                  y_name         = c("pincp", "hicov"),
                  algo_name      = "net")

fits = config |> 
  mutate(fit = pmap(config, 
                    nl$twoway_subset,
                    data     = acs,
                    n_boot   = 0))

fits_obs = fits |> 
  mutate(fit = map(fit, "obs")) |> 
  unnest(fit)


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(fits_obs, here('census', 'temp', 'nonlinear_fits_net.fst'))


# Done --------------------------------------------------------------------
message("Done.")
tictoc::toc()