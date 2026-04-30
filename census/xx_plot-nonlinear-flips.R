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
library(forcats)
library(ggplot2)

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
flips = fits |> 
  filter(algo_name == 'gbt') |>   
  mutate(fit_id = row_number()) |> 
  pivot_longer(cols = c(contains('explained')),
               names_to  = 'stat_name',
               values_to = 'stat_value') |> 
  inner_join(boot_se) |> 
  separate(stat_name, into = c('component', 'ref_pop'), sep = '_') |> 
  arrange(fit_id, component) |> 
  mutate(ref_pop = ifelse(ref_pop == 1, "H", "K")) |> 
  mutate(is_flip = prod(sign(stat_value)) == -1, .by = c(fit_id, component)) |> 
  filter(is_flip)

flips_tidy = flips |> 
  mutate(min_t = min(abs(stat_value / stat_se)),
         max_t = max(abs(stat_value / stat_se)), 
         .by = fit_id) |> 
  arrange(desc(y_name), desc(max_t)) |> 
  mutate(fit_id    = as.integer(fct_inorder(factor(fit_id))),
         component = ifelse(component == 'explained',
                            'Explained Component', 'Unexplained Component'),
         y_name    = ifelse(y_name == 'pincp', 'Log Income', 'Insurance Status'),
         p_value   = 2 * pnorm(-abs(stat_value), sd = stat_se),
         signif    = case_when(p_value < 0.05 ~ 'p < 0.05',
                               p_value < 0.1  ~ 'p < 0.1',
                               TRUE           ~ 'p \u2265 0.1'))

col_pal = c(K = "#1f77b4", H = "#ff7f0e")
shape_pal = c('p < 0.05' = 16,
              'p < 0.1'  = 21,
              'p \u2265 0.1'  = 13)
dodge_width = 0.45
gg = ggplot(flips_tidy,
            aes(x = fit_id,
                y = stat_value,
                color = ref_pop,
                shape = signif)) +
  facet_grid(rows = vars(fct_rev(y_name)),
             cols = vars(component),
             scales = 'free_y',
             switch = 'y') +
  labs(color = "Reference",
       shape = "Significance",
       y     = "Effect Size for...",
       x     = "Experiment ID") +
  theme_bw(base_size = 10) +
  theme(legend.position  = 'top',
        legend.justification = 'left',
        strip.background = element_rect(fill = 'white'), 
        strip.placement  = "outside", 
        axis.text        = element_text(color = 'black'),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank()) +
  guides(color = guide_legend(order = 1),
         shape = guide_legend(order = 2)) +
  scale_x_continuous(breaks = seq(1, 100, 1)) +
  scale_color_manual(values = col_pal) +
  scale_shape_manual(values = shape_pal) +
  geom_hline(yintercept = 0, linetype = 'dashed') +
  geom_linerange(aes(ymin = stat_value - qnorm(0.95) * stat_se,
                     ymax = stat_value + qnorm(0.95) * stat_se),
                 position = position_dodge(width = dodge_width),
                 size = 1.125) +
  geom_linerange(aes(ymin = stat_value - qnorm(0.975) * stat_se,
                     ymax = stat_value + qnorm(0.975) * stat_se),
                 position = position_dodge(width = dodge_width),
                 size = 0.5) +
  geom_point(position = position_dodge(width = dodge_width),
             fill = "white",
             size = 1.75)

ggsave(here('census', 'out', 'nonlinear_flips.pdf'),
       gg,
       width = 8, height = 4, units = "in",
       device = cairo_pdf)
