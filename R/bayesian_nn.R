library(tidyverse)
library(rstan)
library(boot)

relu <- Vectorize(function(x) max(0,x))
log1p_exp <- Vectorize(function(x) log(1+exp(x)))

#create 1000 synthetic samples from log1p_exp network
N <- 1000
synth <- tibble(x1 = runif(N,0,7), x2 = runif(N,0,7)) %>%
  mutate(h1 = log1p_exp((x1+x2)/sqrt(2)), h2 = log1p_exp((x1+x2)/sqrt(2)-5)) %>%
  mutate(y = h1-2*h2) %>%
  mutate(ynoise = y + rnorm(N, 0, 0.1))

synth %>% ggplot(aes(x1,x2, color = y)) + geom_point()

#fit in stan. Try different chains. Different inits etc.
dat <- list(N = N, X = cbind(synth$x1, synth$x2), y = synth$ynoise)
fit <- stan("bayesian_nn.stan", data = dat, chains = 1, iter = 2000, init = list(list(W_raw = matrix(c(1/sqrt(2),1/sqrt(2),1/sqrt(2),1/sqrt(2)),2), w = c(1,-2))))
fit <- stan("bayesian_nn_angle.stan", data = dat, chains = 1, iter = 2000, init = list(list(angles = c(0.25,0.25), w = c(1,-2))))

fit <- stan("bayesian_nn_angle.stan", data = dat, chains = 1, iter = 2000)

fit <- stan("bayesian_nn_angle.stan", data = dat, chains = 1, warmup = 2000, iter = 10002000, init = list(list(angles = c(0.25,0.25), w = c(1,-2))))

fit <- stan("bayesian_nn_angle.stan", data = dat, chains = 1, iter = 10000, init = list(list(angles = c(0.6,-0.75))))

s <- extract(fit)
posterior_draw <- function(s,d) {
  
  theta1 <- atan2(s$W[d,2,1], s$W[d,1,1])/pi
  theta2 <- atan2(s$W[d,2,2], s$W[d,1,2])/pi
  
  list(theta1 = theta1, theta2 = theta2, c = s$c[d,], w = s$w[d,])
}

posterior_draw(s,sample(1:1000,1))

#get log_prob from stan and create the chromosome plot
ll <- function(theta1, theta2) log_prob(fit,unconstrain_pars(fit, list(angles = c(theta1, theta2))))
tibble(theta1 = seq(-0.99,0.99,by=0.01), theta2 = seq(-0.99,0.99,by=0.01)) %>%
  expand(theta1,theta2) %>%
  mutate(l = Vectorize(ll)(theta1, theta2)) %>%
  ggplot(aes(theta1,theta2)) + geom_raster(aes(fill = log(-l)))

#try optimization in stan in the unconstrained and constrained version
m1 <- stan_model("bayesian_nn_angle.stan")
opt <- optimizing(m, data = dat)
opt
r1 <- map(1:1000,function(i) optimizing(m1, data = dat)$value) %>% unlist

m2 <- stan_model("bayesian_nn.stan")
opt <- optimizing(m2, data = dat)
opt
r2 <- map(1:1000,function(i) optimizing(m2, data = dat)$value) %>% unlist

tibble(r1 = r1, r2 = r2) %>% gather(v, val) %>% ggplot(aes(val)) + geom_histogram() + facet_grid(v ~ .)
