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


# Load Data ---------------------------------------------------------------
icu_raw = read_csv(here('Real-data example', 'archive', 'df_clean.csv'))

icu = icu_raw |> 
  mutate(subset_all = 1,
         subset_hr_gt_100  = ifelse(HR > 100, 1, NA),
         subset_map_lt_65  = ifelse(NIMAP < 65, 1, NA),
         subset_temp_gt_38 = ifelse(Temp > 38, 1, NA),
         subset_urine_gt_1000 = ifelse(Urine > 1000, 1, NA))

n_obs = nrow(icu)
n_30  = floor(0.3 * nrow(icu))
n_50  = floor(0.5 * nrow(icu))

for (ii in seq_len(50)) {
  set.seed(ii)
  cn_30 = glue("rand_30_{str_pad(ii, 2, 'left', '0')}")
  cn_50 = glue("rand_50_{str_pad(ii, 2, 'left', '0')}")
  
  icu[[cn_30]] = ifelse(seq_len(n_obs) %in% sample(n_obs, n_30), 1, NA)
  icu[[cn_50]] = ifelse(seq_len(n_obs) %in% sample(n_obs, n_50), 1, NA)
}


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(icu, here('census', 'temp', 'icu_clean.fst'))