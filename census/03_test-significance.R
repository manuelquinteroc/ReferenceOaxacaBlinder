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
fits  = read_fst(here('census', 'temp', 'ob_fits.fst')) |> as_tibble()
boots = read_fst(here('census', 'temp', 'ob_boots.fst')) |> as_tibble()


# Standard Errors ---------------------------------------------------------
boot_se = boots |> 
  pivot_longer(cols = c(starts_with('delta'),
                        ends_with('_0'),
                        ends_with('_1')),
               names_to  = 'stat_name',
               values_to = 'stat_value_boot') |> 
  group_by(subset_name, subset_value, pop_name, x_name, y_name, 
           stat_name) |> 
  summarize(stat_se = sd(stat_value_boot),
            .groups = 'drop')


# Append SEs  -------------------------------------------------------------
flips = fits |> 
  pivot_longer(cols = c(starts_with('delta'),
                        ends_with('_0'),
                        ends_with('_1')),
               names_to  = 'stat_name',
               values_to = 'stat_value_obs') |> 
  inner_join(boot_se) |> 
  mutate(p_val = 2 * pnorm(-abs(stat_value_obs), sd = stat_se))


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