library(MASS) #for Boston housing dataset
library(tidyverse)
library(rstan)
library(parallel)
options(mc.cores = parallel::detectCores())

create_test_train_split <- function(i) {
  bos <- Boston %>% as_tibble %>% mutate_all(function(x) (x-mean(x))/sd(x))
  bos_train <- bos %>% sample_frac(0.8)
  bos_test <- bos %>% anti_join(bos_train)
  
  return(list(train = bos_train, test = bos_test))
}

test_train_splits <- map(1:10, create_test_train_split)

boston_model <- stan_model(file = "stan/boston.stan")
run_boston <- function(test_train_split) {
  
  train <- test_train_split$train
  test <- test_train_split$test
  
  sampling(boston_model, chains = 1, iter = 1000, refresh = 10,
       data = list(N = nrow(train), D = ncol(train)-1,
                   X = select(train,-medv) %>% as.matrix,
                   y = train$medv, N_test = nrow(test),
                   X_test = select(test,-medv) %>% as.matrix),
       init = list(list(theta_principal = runif(50,-pi,pi),
                        theta_lower = matrix(runif(50*(12-1),-pi/2,pi/2),nrow=50),
                        beta = rnorm(50), c_raw = sort(rnorm(50)))))
}

boston_ordered_model <- stan_model(file = "stan/boston_ordered.stan")
run_boston_ordered <- function(test_train_split) {
  
  train <- test_train_split$train
  test <- test_train_split$test
  
  sampling(boston_ordered_model, chains = 1, iter = 1000, refresh = 10,
       data = list(N = nrow(train), D = ncol(train)-1,
                   X = select(train,-medv) %>% as.matrix,
                   y = train$medv, N_test = nrow(test),
                   X_test = select(test,-medv) %>% as.matrix))
}

boston_unconstrained_model <- stan_model(file = "stan/boson_unconstrained.stan")
run_boston_unconstrained <- function(test_train_split) {
  
  train <- test_train_split$train
  test <- test_train_split$test
  
  sampling(boston_unconstrained_model, chains = 1, iter = 1000, refresh = 10,
       data = list(N = nrow(train), D = ncol(train)-1,
                   X = select(train,-medv) %>% as.matrix,
                   y = train$medv, N_test = nrow(test),
                   X_test = select(test,-medv) %>% as.matrix))
}

#simaltaneously fit the model to each of the 10 test train splits
fits_boston <- mclapply(test_train_splits,run_boston)
fits_boston_ordered <- mclapply(test_train_splits,run_boston_ordered)
fits_boston_unconstrained <- mclapply(test_train_splits,run_boston_unconstrained)

stan_fit_to_summary <- function(fit, test_train_split) {
  s <- extract(fit)
  q2 <- s$yhat %>% apply(2, function(x) quantile(x,probs = c(0.025)))
  q97 <- s$yhat %>% apply(2, function(x) quantile(x,probs = c(0.975)))
  q50 <- s$yhat %>% apply(2, function(x) quantile(x,probs = c(0.5)))
  yhat <- tibble(q2 = q2, q50 = q50, q97 = q97, y = test_train_split$test$medv) %>%
    mutate(err = (y-q50)^2) %>%
    mutate(outlier = q2 <= y & y <= q97)
  
  yhat %>% summarize(sse = sqrt(sum(err)/101), pct_outliers = mean(!outlier))
}

map2(fits_boston, test_train_splits, stan_fit_to_summary) %>% bind_rows
map2(fits_boston_ordered, test_train_splits, stan_fit_to_summary) %>% bind_rows
map2(fits_boston_unconstrained, test_train_splits, stan_fit_to_summary) %>% bind_rows

m <- stan_model(file = "stan/boston.stan")
v <- vb(m, data = list(N = nrow(bos_train), D = ncol(bos_train)-1, X = select(bos_train,-medv) %>% as.matrix, y = bos_train$medv, N_test = nrow(bos_test), X_test = select(bos_test,-medv) %>% as.matrix),
        init = list(theta_principal = matrix(runif(50,-pi,pi),nrow=50), theta_lower = matrix(runif(50*(13-1),-pi/2,pi/2),nrow=50),
                    beta = rnorm(50), c_raw = sort(rnorm(50))))


m <- stan_model(file = "stan/boson_multi_layer.stan")
v <- vb(m, data = list(N = nrow(bos_train), D = ncol(bos_train)-1, X = select(bos_train,-medv) %>% as.matrix, y = bos_train$medv, N_test = nrow(bos_test), X_test = select(bos_test,-medv) %>% as.matrix),
        init = list(theta_principal = matrix(runif(50,-pi,pi),nrow=50), theta_lower = matrix(runif(50*(13-1),-pi/2,pi/2),nrow=50),
                    beta = rnorm(50), c_raw = sort(rnorm(50))))

m <- stan_model(file = "stan/boson_unconstrained.stan")
v_unconstrained <- vb(m, data = list(N = nrow(bos_train), D = ncol(bos_train)-1, X = select(bos_train,-medv) %>% as.matrix, y = bos_train$medv, N_test = nrow(bos_test), X_test = select(bos_test,-medv) %>% as.matrix))
