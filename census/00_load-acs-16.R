# -------------------------------------------------------------------------
# Extract variables and enforce sample restrictions
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

# Load Data ---------------------------------------------------------------
load_acs = function() {
  
  acs_cols = cols_only(ST    = 'i',  # State
                       AGEP  = 'i',  # Age
                       ESR   = 'i',  # Employment status
                       WKHP  = 'i',  # Hours worked per week
                       WKW   = 'i',  # Weeks worked per year (categorical)
                       PERNP = 'i',  # Earnings
                       PINCP = 'i',  # Income
                       NATIVITY = 'i', # Decade of Entry
                       HICOV    = 'i', # Health Insurance Coverage
                       SEX      = 'i',  # Sex
                       RAC1P    = 'i',  # Race
                       SCHL     = 'i',  # Educational attainment
                       MAR      = 'i',  # Marital Status
                       INDP     = 'i', # Industry
                       NAICSP   = 'c'  # Industry
  )
  
  acs_a = read_csv(here('census', 'raw', 'ss16pusa.csv'), 
                   col_types = acs_cols)
  acs_b = read_csv(here('census', 'raw', 'ss16pusb.csv'), 
                   col_types = acs_cols)
  
  acs = bind_rows(acs_a, acs_b)
  acs |> 
    rename_with(.cols = everything(), .fn = tolower) |> 
    na.omit()
}

acs_raw = load_acs()

acs = acs_raw |> 
  tidylog::filter(between(agep, 25, 65),
                  wkhp >= 35,
                  wkw == 1, # 50-52 weeks per year
                  esr == 1, # Civilian employed, at work
                  pernp >= 12687.50,
                  pincp >= 12687.50) |> 
  transmute(st,
            indp_2     = substr(indp, 1, 2),
            naics_2    = substr(naicsp, 1, 2),
            naics_3    = substr(naicsp, 1, 3),
            pincp      = log(pincp),
            sex_female = sex == 2,
            hicov      = hicov == 1,
            immigrant  = nativity == 2,
            mar        = mar == 1,
            educ_bach  = schl >= 21,
            race_bw = case_when(rac1p == 1 ~ 0,
                                rac1p == 2 ~ 1,
                                TRUE ~ NA_real_)) |> 
  mutate(across(where(is.logical), as.numeric))


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(acs, here('census', 'temp', 'acs16_workforce.fst'))


# Done --------------------------------------------------------------------
message("Done.")

