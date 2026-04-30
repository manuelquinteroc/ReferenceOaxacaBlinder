twoway_subset = function(data, y_name, pop_name, subset_name, subset_value, algo_name, 
                         x_names = NULL, n_boot = 1000,
                         verbose_outer = F, verbose_inner = F, parallel_inner = T) {
  
  
  if (verbose_outer) {
    tictoc::tic()
    message("algo_name: ", algo_name,
            " | subset_name: ", subset_name,
            " | subset_value ", subset_value,
            " | y_name: ", y_name,
            " | pop_name: ", pop_name)  
  }
  
  
  # Restrict to the subset in question  
  data = data[ as.character(data[[subset_name]]) == as.character(subset_value), ]
  data = data[ !is.na(data[[pop_name]]), ]
  
  # Choose the algorithm by name
  algo = switch(algo_name,
                'gbt' = gbt,
                'ols' = ols,
                'glm' = glm,
                'net' = net,
                stop("Unrecognized algo_name: ", algo_name))
  
  # Compute the OBD with both ref. pops on observed data
  data_prepped = prepare_data(data, pop_name, y_name, x_names)
  obs = decompose_mean(data_prepped, algo)
  
  # Recompute OBD on bootstrapped data
  if (parallel_inner) {
    boot = furrr::future_map(seq_len(n_boot),
                             \(seed) {
                               if (verbose_inner) message("Bootstrap iter: ", seed)
                               set.seed(seed)
                               data_boot    = resample(data, pop_name)
                               data_prepped = prepare_data(data_boot, pop_name, y_name, x_names)
                               decompose_mean(data_prepped, algo)
                             },
                             .options = furrr::furrr_options(seed = 1)) 
  } else {
    boot = map(seq_len(n_boot),
               \(seed) {
                 if (verbose_inner) message("Bootstrap iter: ", seed)
                 set.seed(seed)
                 data_boot    = resample(data, pop_name)
                 data_prepped = prepare_data(data_boot, pop_name, y_name, x_names)
                 decompose_mean(data_prepped, algo)
               })
  }
  boot = purrr::list_rbind(boot, names_to = "boot_iter")
  
  if (verbose_outer) { tictoc::toc() }
  
  # Return
  list(obs  = obs,
       boot = boot)
}

prepare_data = function(data, pop_name, y_name, x_names = NULL) {
  
  if (is_null(x_names)) {
    features = data |> 
      select(-all_of(c(pop_name, y_name, "naics_2"))) |> 
      dummy_cols(select_columns          = setdiff(c("st", "indp_2"),
                                                   c(pop_name, y_name)),
                 remove_selected_columns = TRUE) |> 
      as.matrix()
    
  } else {
    features = data |> 
      select(all_of(x_names)) |> 
      as.matrix()
  }
  
  # add dummy for missing values
  for (ii in seq_len(ncol(features))) {
    if (any(is.na(features[,ii]))) {
      features      = cbind(features, is.na(features[,ii]))
      features[,ii] = replace_na(is.na(features[,ii]), 0)
      colnames(features)[length(colnames(features))] = paste0("is_na_", ii)
    }
  }
  
  list(pop      = data[[pop_name]], 
       y        = data[[y_name]],
       features = features,
       folds    = assign_folds(data[[pop_name]], data[[y_name]]))
}

# Stratified cross-validation folds
assign_folds = function(pop, y, seed = 1) {
  
  set.seed(seed)
  
  # cut continuous outcomes into quintiles for stratified CV
  if (any((y %in% 0:1) == FALSE) ) { y = dplyr::ntile(y, 5) }
  
  fold_template = c(1:3)
  
  fold_tb = tibble(pop_value  = pop, 
                   y_value = y) |> 
    group_by(pop_value, y_value) |> 
    mutate(fold = rep(fold_template, length.out = n()),
           fold = sample(fold))
  
  fold_tb$fold
  
}

# OBD-style decomp of explained and unexplained components
decompose_mean = function(data, algo) {
  
  fit_0 = algo(x     = data$features[data$pop == 0, ],
               y     = data$y[data$pop == 0],
               folds = data$folds[data$pop == 0])
  
  fit_1 = algo(x     = data$features[data$pop == 1, ],
               y     = data$y[data$pop == 1],
               folds = data$folds[data$pop == 1])
  
  mu_x0_yx_0 = predict_mean(fit_0, data$features[data$pop == 0, ])
  mu_x1_yx_0 = predict_mean(fit_0, data$features[data$pop == 1, ])
  
  mu_x0_yx_1 = predict_mean(fit_1, data$features[data$pop == 0, ])
  mu_x1_yx_1 = predict_mean(fit_1, data$features[data$pop == 1, ])
  
  tibble(delta_y       = with(data, mean(y[pop == 1]) - mean(y[pop == 0])),
         explained_1   = mu_x1_yx_1 - mu_x0_yx_1,
         unexplained_1 = mu_x0_yx_1 - mu_x0_yx_0,
         explained_0   = mu_x1_yx_0 - mu_x0_yx_0,
         unexplained_0 = mu_x1_yx_1 - mu_x1_yx_0,
         n_1           = sum(data$pop == 1),
         n_0           = sum(data$pop == 0))
}

aligned_slope_intercept = function(data, y_name, pop_name, subset_name, subset_value) {
  
  # Restrict to the subset in question  
  data = data[ as.character(data[[subset_name]]) == as.character(subset_value), ]
  data = data[ !is.na(data[[pop_name]]), ]
  data = prepare_data(data, pop_name, y_name)
  
  fit_0 = ols(x     = data$features[data$pop == 0, ],
              y     = data$y[data$pop == 0],
              folds = data$folds[data$pop == 0])
  
  fit_1 = ols(x     = data$features[data$pop == 1, ],
              y     = data$y[data$pop == 1],
              folds = data$folds[data$pop == 1])
  
  mu_x0_yx_0 = predict_mean(fit_0, data$features[data$pop == 0, ])
  mu_x1_yx_0 = predict_mean(fit_0, data$features[data$pop == 1, ])
  
  mu_x0_yx_1 = predict_mean(fit_1, data$features[data$pop == 0, ])
  mu_x1_yx_1 = predict_mean(fit_1, data$features[data$pop == 1, ])
  
  explained_1   = mu_x1_yx_1 - mu_x0_yx_1
  explained_0   = mu_x1_yx_0 - mu_x0_yx_0
  
  delta_alpha = coef(fit_1$fit)[1] - coef(fit_0$fit)[1]
  
  # check condition
  (explained_0 * explained_1 > 0) &
    (sign(delta_alpha) == sign(explained_0)) &
    (sign(delta_alpha) == sign(explained_1))
  
}

predict_mean = function(model, x) {
  
  y_hat = switch(class(model$fit)[1],
                 lgb.Booster = predict(model$fit, x, type = 'response'),
                 glm         = predict_glm(model, x),
                 brulee_mlp  = predict_brl(model, x),
                 predict(model$fit, x))
  
  mean(as.numeric(y_hat))
}

# Tune and train gradient-boosted trees
gbt = function(x, y, folds) {
  
  data_tune = lgb.Dataset(data  = x[folds != 0, ],
                          label = y[folds != 0])
  
  folds_tune      = folds[folds != 0]
  folds_tune_list = map(unique(folds_tune), \(ff) { which(folds_tune == ff)})
  
  y_type = if (all(y %in% 0:1)) 'binary' else 'continuous'
  
  lgb_params = list(objective      = switch(y_type, binary = 'binary', continuous = 'regression'),
                    learning_rate  = 0.25,
                    num_iterations = 10000L,
                    num_leaves     = 7L,
                    verbose        = -1,
                    num_threads    = 1L)
  
  tuned = lgb.cv(data   = data_tune, 
                 params = lgb_params,
                 folds  = folds_tune_list,
                 early_stopping_rounds = 10)
  
  data_train = lgb.Dataset(data  = x, label = y)
  
  list(y_type = y_type,
       fit    = lightgbm(data_train, 
                         params  = lgb_params, 
                         nrounds = tuned$best_iter))
  
}

ridge = function(x, y, folds) {
  tuned = cv.glmnet(x[folds != 0, ], y = y[folds != 0], foldid = folds[folds != 0], alpha = 0)
  fit    = glmnet(x[folds == 0, ], y[folds == 0], alpha = 0, lambda = tuned$lambda)
  fit$lambda.pred = tuned$lambda.min
  return(fit)
}

ols = function(x, y, folds) {
  
  y_type = 'continuous'
  
  fit = glm.fit(x = cbind(1, x[folds != 0,]),
                y = y[folds != 0], 
                family = gaussian())
  
  class(fit) = c('glm', 'list')
  
  list(y_type = y_type,
       fit    = fit)
  
}

glm = function(x, y, folds) {
  
  y_type = if (all(y %in% 0:1)) 'binary' else 'continuous'
  
  fit = glm.fit(x = cbind(1, x[folds != 0,]),
                y = y[folds != 0], 
                family = switch(y_type, binary = binomial(), continuous = gaussian()))
  
  class(fit) = c('glm', 'list')
  
  list(y_type = y_type,
       fit    = fit)
}

net = function(x, y, folds) {
  
  y_type = if (all(y %in% 0:1)) 'binary' else 'continuous'
  
  fit = brulee::brulee_mlp(x, 
                           switch(y_type, continuous = y, binary = factor(y)), 
                           hidden_units = c(16, 8),
                           optimizer    = "SGD",
                           batch_size   = 512,
                           validation   = 0.2,
                           penalty      = 10^-4,
                           activation   = 'relu')
  
  list(y_type = y_type,
       fit    = fit)
}

predict_glm = function(model, x) {
  
  cf = model$fit$coefficients
  cf = cf[!is.na(cf)]
  
  cf_names = names(cf)
  cf_names = cf_names[cf_names != ""]
  
  eta = as.numeric(cbind(1, x[,cf_names]) %*% cf)
  
  if (model$y_type == 'binary') {
    plogis(eta)
  } else if (model$y_type == 'continuous') {
    eta
  }
}

predict_brl = function(model, x) {
  y_hat = switch(model$y_type,
                 continuous = predict(model$fit, x)[,1],
                 binary     = predict(model$fit, x, type = 'prob')[,2])
  unlist(y_hat, use.names = F)
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