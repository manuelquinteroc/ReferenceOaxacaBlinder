# -------------------------------------------------------------------------
# Hypothesis testing based on bootstrap
# -------------------------------------------------------------------------

# Libraries ---------------------------------------------------------------
library(here)
library(fst)
library(readr)
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)

# Read Data ---------------------------------------------------------------
fits  = read_fst(here('census', 'temp', 'nonlinear_fits.fst')) |> as_tibble()
boots = read_fst(here('census', 'temp', 'nonlinear_boots.fst')) |> as_tibble()


# Standard Errors ---------------------------------------------------------
boot_se = boots |> 
  pivot_longer(cols = c(starts_with('delta'),
                        ends_with('_0'),
                        ends_with('_1')),
               names_to  = 'stat_name',
               values_to = 'stat_value_boot') |> 
  group_by(subset_name, subset_value, pop_name, algo_name, y_name, 
           stat_name) |> 
  summarize(stat_se = sd(stat_value_boot),
            .groups = 'drop')


# Append SEs  -------------------------------------------------------------
flips_components = fits |> 
  filter(n_0 > 50, n_1 > 50, abs(delta_y) > 0.01) |> 
  mutate(fit_id = row_number()) |> 
  pivot_longer(cols = c(contains('explained')),
               names_to  = 'stat_name',
               values_to = 'stat_value') |> 
  left_join(boot_se) |> 
  separate(stat_name, into = c('component', 'ref_pop'), sep = '_') |> 
  mutate(p_value = 2 * pnorm(-abs(stat_value), sd = stat_se)) |> 
  group_by(subset_name, subset_value, pop_name, y_name, algo_name, component) |> 
  summarize(is_flip     = prod(stat_value) < 0,
            min_p_value = min(p_value),
            .groups = 'drop') |> 
  mutate(reject_10 = is_flip & (min_p_value < 0.10),
         reject_05 = is_flip & (min_p_value < 0.05),
         reject_01 = is_flip & (min_p_value < 0.01))
  
flip_counts_component = flips_components |> 
  group_by(y_name, algo_name, component) |> 
  summarize(n_fit     = n(),
            is_flip   = sum(is_flip),
            reject_10 = sum(reject_10, na.rm = T),
            reject_05 = sum(reject_05, na.rm = T),
            reject_01 = sum(reject_01, na.rm = T),
            .groups = 'drop') |> 
  arrange(desc(y_name), desc(algo_name), component)


flips_total = flips_components |> 
  group_by(subset_name, subset_value, pop_name, y_name, algo_name) |> 
  summarize(is_flip = any(is_flip),
            reject_10 = any(reject_10),
            reject_05 = any(reject_05),
            reject_01 = any(reject_01),
            .groups = 'drop')

flip_counts_total = flips_total |> 
  group_by(y_name, algo_name) |> 
  summarize(component = "either", 
            n_fit     = n(),
            is_flip   = sum(is_flip),
            reject_10 = sum(reject_10, na.rm = T),
            reject_05 = sum(reject_05, na.rm = T),
            reject_01 = sum(reject_01, na.rm = T),
            .groups = 'drop') |> 
  arrange(desc(y_name), desc(algo_name))

flip_counts = bind_rows(flip_counts_component,
                        flip_counts_total)

write_csv(flip_counts, here('census', 'out', 'flip_counts.csv'))
