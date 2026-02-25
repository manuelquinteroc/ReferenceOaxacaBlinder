# Libraries ---------------------------------------------------------------
renv::activate()
library(here)
library(tidyr)
library(dplyr)
library(purrr)
library(glue)
library(ggplot2)

# Helpers -----------------------------------------------------------------
# Irwin-Hall CDF (approximate for large n)
pirwinhall = function(x, n) {
  if (n > 80) { pirwinhall_approx(x, n) } else { pirwinhall_exact(x, n) }
}

pirwinhall_exact = function(x, n) {
  
  sapply(x, 
         \(xx) {
           k = seq(0, floor(xx))
           sum((-1)**k * choose(n, k) * (xx - k)**n) / factorial(n)
         })
}

# Gaussian approximation to Irwin-Hall distribution
pirwinhall_approx = function(x, n) { pnorm(x, mean = n / 2, sd = sqrt(n / 12)) }

# Irwin-Hall PDF (approximate for large n)
dirwinhall = function(x, n) {
  if (n > 50) {
    dirwinhall_approx(x, n)
  } else {
    dirwinhall_exact(x, n)
  }
}

dirwinhall_exact = function(x, n) {
  
  sapply(x, 
         \(xx) {
           k = seq(0, floor(xx))
           sum((-1)**k * choose(n, k) * (xx - k)**{n-1}) / factorial(n-1)
         })
}

dirwinhall_approx = function(x, n) {
  dnorm(x, mean = n / 2, sd = sqrt(n / 12))
}

# Sign flip is probability for falling in either tail
pr_signflip = function(d, m) {
  
  lower = integrate(\(i) (1 - pirwinhall(x = d + 1 - i, n = 2*d)) * dirwinhall(x = i, n = 2),
                    lower = 0,
                    upper = 1)$value
  
  upper = integrate(\(i) pirwinhall(x = d + 1 - i, n = 2*d) * dirwinhall(x = i, n = 2),
                    lower = 1,
                    upper = 2)$value
  
  lower + upper
}


# Setup --------------------------------------------------------------------
design = tidyr::crossing(d = 1:100,
                         m = 10)
gg_tb = dplyr::mutate(design,
                      pr_unexplained = purrr::pmap_dbl(design, pr_signflip),
                      pr_explained   = 0.5) |> 
  pivot_longer(c(pr_explained, pr_unexplained),
               names_prefix = 'pr_',
               names_to = 'component',
               values_to = 'pr') |> 
  mutate(component = ifelse(component == 'explained',
                            'Explained Component',
                            'Unexplained Component'))


# Plot ---------------------------------------------------------------------
gg = ggplot(gg_tb,
            aes(x = d, y = pr, color = component)) +
  labs(x     = "Dimensionality of Covariates (d)",
       y     = "Percentage of Parameter Space",
       color = "Sign Flip in...") +
  theme_bw(base_size = 28) +
  theme(axis.text       = element_text(color = 'black'),
        legend.position = "inside",
        legend.position.inside = c(0.7, 0.2),
        legend.background = element_rect(colour = "black")) +
  scale_color_manual(values = c('Explained Component'   = '#785EF0',
                                'Unexplained Component' = '#FE6100')) +
  scale_y_continuous(labels = scales::percent) +
  geom_line(linewidth = 2)

ggsave(here('sign_flip_probability', 'out', 'standardized.pdf'),
       gg, 
       width = 9, height = 6.5, units = "in")
