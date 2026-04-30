# Compute OBD (observed and bootstrapped) on a subset of data
twoway_subset = function(data, x_name, y_name, pop_name, subset_name, subset_value, n_boot = 1000) {
  
  # Restrict to the subset in question  
  data = data[ as.character(data[[subset_name]]) == as.character(subset_value), ]
  data = data[ !is.na(data[[pop_name]]), ]
  
  # Compute the OBD with both ref. groups on observed data
  obs = twoway(x = data[[x_name]],
               y = data[[y_name]],
               pop = data[[pop_name]])
  
  # Recompute OBD on bootstrapped data
  boot = furrr::future_map(seq_len(n_boot),
                           \(seed) {
                             set.seed(seed)
                             data_boot = resample(data, pop_name)
                             twoway(x = data_boot[[x_name]],
                                    y = data_boot[[y_name]],
                                    pop = data_boot[[pop_name]])
                           },
                           .options = furrr::furrr_options(seed = 1))
  boot = purrr::list_rbind(boot, names_to = "boot_iter")
  
  list(obs  = obs,
       boot = boot)
}

# Oaxaca-Blinder decomposition with both reference groups
twoway = function(x, y, pop) {
  # Notation: H = 1, K = 0
  
  # fast regression with lm.fit
  fit_0 = lm.fit(cbind(1, x[pop == 0]), y[pop == 0])
  fit_1 = lm.fit(cbind(1, x[pop == 1]), y[pop == 1])
  
  x_bar = c(mean(x[pop == 0]), mean(x[pop == 1]))
  y_bar = c(mean(y[pop == 0]), mean(y[pop == 1]))
  
  # compute R2 for both fihts
  ssr = c(sum(fit_0$residuals^2), sum(fit_1$residuals^2))
  sst = c(sum((y[pop == 0] - y_bar[1])^2), 
          sum((y[pop == 1] - y_bar[2])^2))
  r2  = 1 - ssr/sst
  
  # predicted ("counterfactual") means
  y_hat_0 = c(fit_0$coef[1] + fit_0$coef[2] * x_bar[1],
              fit_0$coef[1] + fit_0$coef[2] * x_bar[2])
  
  y_hat_1 = c(fit_1$coef[1] + fit_1$coef[2] * x_bar[1],
              fit_1$coef[1] + fit_1$coef[2] * x_bar[2])
  
  tibble(ref_value   = 0:1,
         n           = c(sum(pop == 0), sum(pop == 1)),
         delta_x     = diff(x_bar),
         delta_alpha = fit_1$coef[1] - fit_0$coef[1],
         delta_beta  = fit_1$coef[2] - fit_0$coef[2],
         delta_y     = diff(y_bar),
         explained   = c(diff(y_hat_0), diff(y_hat_1)),
         unexplained = rev(y_hat_1 - y_hat_0),
         r2          = r2)
}

# resample observations within strata
resample = function(tb, strata_name) {
  
  ids = split(seq_len(nrow(tb)), tb[[strata_name]])
  ids = lapply(ids, \(x) {
    
    if (length(x) == 1) return(x)
    sample(x, size = length(x), replace = TRUE)
    
  } )
  
  tb[unlist(ids, use.names = FALSE), ]
}
