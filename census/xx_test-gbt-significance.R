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
library(scales)

# Read Data ---------------------------------------------------------------
fits  = read_fst(here('census', 'temp', 'icu_gbt_fits.fst')) |> as_tibble()
boots = list.files(here('census', 'temp', 'icu_gbt_boots'), full.names = T) |> 
  map(read_fst) |> 
  bind_rows() |> 
  as_tibble()

read_fst(here('census', 'temp', 'ob_boots.fst')) |> as_tibble()


# Standard Errors ---------------------------------------------------------
boot_se = boots |> 
  pivot_longer(cols = c(ends_with('_0'),
                        ends_with('_1')),
               names_to  = 'stat_name',
               values_to = 'stat_value_boot') |> 
  group_by(subset_name, subset_value, pop_name, algo_name, y_name, 
           stat_name) |> 
  summarize(se = sd(stat_value_boot),
            .groups = 'drop')


# Append SEs  -------------------------------------------------------------
flips = fits |> 
  mutate(fit_id = row_number()) |> 
  select(-starts_with('n_')) |> 
  pivot_longer(cols = c(ends_with('_0'),
                        ends_with('_1')),
               names_to  = 'stat_name',
               values_to = 'value') |> 
  inner_join(boot_se) |> 
  mutate(p = 2 * pnorm(-abs(value), sd = se))

flips_wide = flips |> 
  select(-se) |> 
  separate(stat_name, c('component', 'pop'), sep = '_') |> 
  pivot_longer(c(value, p), 
               names_to = 'stat_name', 
               values_to = 'stat_value') |> 
  pivot_wider(names_from = c(component, stat_name, pop),
              values_from = stat_value)

thresh = 0.05
flips_wide |> 
  filter(sign(explained_value_0 * explained_value_1) == -1 & pmin(explained_p_0, explained_p_1) < thresh | 
           sign(unexplained_value_0 * unexplained_value_1) == -1 & pmin(unexplained_p_0, unexplained_p_1) < thresh)

# Format for latex
flip_tb = flips |> 
  filter(str_detect(stat_name, 'explained') | stat_name == 'delta_y') |> 
  mutate(stat_value_obs = scales::number(stat_value_obs, accuracy = 0.001),
         stat_se        = scales::number(stat_se, accuracy = 0.001),
         p_val          = scales::pvalue(p_val)) |> 
  pivot_longer(c(stat_value_obs, stat_se, p_val)) |> 
  pivot_wider(names_from = stat_name, values_from = value)

n_tb = flips |> 
  filter(stat_name %in% c('n_0', 'n_1')) |> 
  group_by(subset_name, subset_value, pop_name, x_name, y_name) |> 
  summarize(name = 'stat_value_obs',
            n    = sum(stat_value_obs),
            .groups = 'drop')

out = left_join(flip_tb, n_tb) |> 
  relocate(n, .before = delta_y) |> 
  relocate(explained_0, unexplained_0, .after = everything())

# copy to LaTeX (it's just one table)
write_csv(out, here('census', 'out', 'flip_signif.csv'))