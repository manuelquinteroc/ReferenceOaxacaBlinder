# -------------------------------------------------------------------------
# Extract variables and enforce sample restrictions
# -------------------------------------------------------------------------

# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)

# Load Data ---------------------------------------------------------------
load_acs = function() {

  # Read only the columns needed; INDP is integer-coded, NAICSP is the raw NAICS string
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
  # Sample restrictions from Bach et al. (2024)
  tidylog::filter(between(agep, 25, 65),    # prime working age
                  wkhp >= 35,               # full-time (≥35 hrs/week)
                  wkw == 1,                 # full-year (50–52 weeks)
                  esr == 1,                 # civilian employed, at work (excludes self-employed & unemployed)
                  pernp >= 12687.50,        # ≥ full-time federal minimum wage ($7.25 × 35 hrs × 50 wks)
                  pincp >= 12687.50) |>     # same floor on total personal income
  transmute(st,
            indp_2     = substr(indp, 1, 2),    # 2-digit INDP code used as subset identifier
            naics_2    = substr(naicsp, 1, 2),  # 2-digit NAICS used as subset identifier
            naics_3    = substr(naicsp, 1, 3),
            pincp      = log(pincp),            # log income as continuous outcome
            sex_female = sex == 2,
            hicov      = hicov == 1,
            immigrant  = nativity == 2,
            mar        = mar == 1,
            educ_bach  = schl >= 21,            # bachelor's degree or higher
            # NA for non-Black/non-White respondents; those rows are still used for
            # decompositions where race_bw is not the population variable
            race_bw = case_when(rac1p == 1 ~ 0,
                                rac1p == 2 ~ 1,
                                TRUE ~ NA_real_)) |>
  mutate(across(where(is.logical), as.numeric))


# Save --------------------------------------------------------------------
message("Saving...")
write_fst(acs, here('census', 'temp', 'acs16_workforce.fst'))


# Done --------------------------------------------------------------------
message("Done.")

