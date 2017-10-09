library(MASS) #for Boston housing dataset
library(tidyverse)
library(rstan)

bos <- Boston %>% as_tibble %>% mutate_all(function(x) (x-mean(x))/sd(x))
bos_train <- bos %>% sample_frac(0.8)
bos_test <- bos %>% anti_join(bos_train)

run_boston <- function(i) {
  stan(file = "stan/boston.stan", chains = 1, iter = 1000, refresh = 10,
       data = list(N = nrow(bos_train), D = ncol(bos_train)-1,
                   X = select(bos_train,-medv) %>% as.matrix,
                   y = bos_train$medv, N_test = nrow(bos_test),
                   X_test = select(bos_test,-medv) %>% as.matrix),
       init = list(list(theta_principal = runif(50,-pi,pi),
                        theta_lower = matrix(runif(50*(12-1),-pi/2,pi/2),nrow=50),
                        beta = rnorm(50), c_raw = sort(rnorm(50)))))
}

run_boston_ordered <- function(i) {
  stan(file = "stan/boston_ordered.stan", chains = 1, iter = 1000, refresh = 10,
       data = list(N = nrow(bos_train), D = ncol(bos_train)-1,
                   X = select(bos_train,-medv) %>% as.matrix,
                   y = bos_train$medv, N_test = nrow(bos_test),
                   X_test = select(bos_test,-medv) %>% as.matrix))
}

run_boston_unconstrained <- function(i) {
  stan(file = "stan/boson_unconstrained.stan", chains = 1, iter = 1000, refresh = 10,
       data = list(N = nrow(bos_train), D = ncol(bos_train)-1,
                   X = select(bos_train,-medv) %>% as.matrix,
                   y = bos_train$medv, N_test = nrow(bos_test),
                   X_test = select(bos_test,-medv) %>% as.matrix))
}

library(parallel)
fits_boston <- mclapply(1:10,run_boston)

fit <- stan(file = "stan/boston.stan", chains = 1, iter = 1000, refresh = 10,
            data = list(N = nrow(bos_train), D = ncol(bos_train)-1,
                        X = select(bos_train,-medv) %>% as.matrix,
                        y = bos_train$medv, N_test = nrow(bos_test),
                        X_test = select(bos_test,-medv) %>% as.matrix),
            init = list(list(theta_principal = runif(50,-pi,pi),
                             theta_lower = matrix(runif(50*(12-1),-pi/2,pi/2),nrow=50),
                             beta = rnorm(50), c_raw = sort(rnorm(50)))))

fit_unconstrained <- stan(file = "stan/boson_unconstrained.stan", chains = 1, iter = 1000, refresh = 10,
                          data = list(N = nrow(bos_train), D = ncol(bos_train)-1,
                                      X = select(bos_train,-medv) %>% as.matrix,
                                      y = bos_train$medv, N_test = nrow(bos_test),
                                      X_test = select(bos_test,-medv) %>% as.matrix))

m <- stan_model(file = "stan/boston.stan")
v <- vb(m, data = list(N = nrow(bos_train), D = ncol(bos_train)-1, X = select(bos_train,-medv) %>% as.matrix, y = bos_train$medv, N_test = nrow(bos_test), X_test = select(bos_test,-medv) %>% as.matrix),
        init = list(theta_principal = matrix(runif(50,-pi,pi),nrow=50), theta_lower = matrix(runif(50*(13-1),-pi/2,pi/2),nrow=50),
                         beta = rnorm(50), c_raw = sort(rnorm(50))))

s <- extract(fit_unconstrained)
q2 <- s$yhat %>% apply(2, function(x) quantile(x,probs = c(0.025)))
q97 <- s$yhat %>% apply(2, function(x) quantile(x,probs = c(0.975)))
q50 <- s$yhat %>% apply(2, function(x) quantile(x,probs = c(0.5)))
yhat <- tibble(q2 = q2, q50 = q50, q97 = q97, y = bos_test$medv) %>%
  mutate(err = (y-q50)^2) %>%
  mutate(outlier = q2 <= y & y <= q97)

yhat %>% summarize(sse = sqrt(sum(err)/101), pct_outliers = mean(!outlier))

m <- stan_model(file = "stan/boson_multi_layer.stan")
v <- vb(m, data = list(N = nrow(bos_train), D = ncol(bos_train)-1, X = select(bos_train,-medv) %>% as.matrix, y = bos_train$medv, N_test = nrow(bos_test), X_test = select(bos_test,-medv) %>% as.matrix),
        init = list(theta_principal = matrix(runif(50,-pi,pi),nrow=50), theta_lower = matrix(runif(50*(13-1),-pi/2,pi/2),nrow=50),
                    beta = rnorm(50), c_raw = sort(rnorm(50))))

m <- stan_model(file = "stan/boson_unconstrained.stan")
v_unconstrained <- vb(m, data = list(N = nrow(bos_train), D = ncol(bos_train)-1, X = select(bos_train,-medv) %>% as.matrix, y = bos_train$medv, N_test = nrow(bos_test), X_test = select(bos_test,-medv) %>% as.matrix))
