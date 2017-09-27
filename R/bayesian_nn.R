library(tidyverse)
library(rstan)
library(shinystan)
library(boot)
options(mc.cores = parallel::detectCores())

relu <- Vectorize(function(x) max(0,x))
log1p_exp <- Vectorize(function(x) log(1+exp(x)))

#create 1000 synthetic samples from log1p_exp network
N <- 1000
synth <- tibble(x1 = runif(N,-10,10), x2 = runif(N,-10,10)) %>%
  mutate(h1 = relu((x1+x2)/sqrt(2)), h2 = relu((-x1+x2)/sqrt(2)-5)) %>%
  mutate(y = h1-2*h2) %>%
  mutate(ynoise = y + rnorm(N, 0, 0.1))

synth %>% ggplot(aes(x1,x2, color = y)) + geom_point()

fit <- stan("stan/bayesian_nn.stan", data = dat, chains = 1, iter = 2000, init = list(list(W_raw = matrix(c(1/sqrt(2),-1/sqrt(2),1/sqrt(2),1/sqrt(2)),2), c_raw = c(0,5), w = c(1,-2))))
fit <- stan("stan/bayesian_nn_stiefel.stan", data = dat, chains = 1, iter = 2000, init = list(list(theta1_principal = as.array(pi/4), theta1_lower = as.array(0), theta2_principal = as.array(3*pi/4), theta2_lower = as.array(0), beta = c(1,-2), c_raw = c(0,5))))

fit <- stan("stan/bayesian_nn.stan", data = dat, chains = 1, iter = 2000, init = list(list(W_raw = matrix(rnorm(4),2), c_raw = sort(rnorm(2)), w = rnorm(2))))
fit <- stan("stan/bayesian_nn_stiefel.stan", data = dat, chains = 1, iter = 2000, init = list(list(theta1_principal = as.array(runif(1,-pi,pi)), theta1_lower = as.array(0), theta2_principal = as.array(runif(1,-pi,pi)), theta2_lower = as.array(0), beta = rnorm(2), c_raw = sort(rnorm(2)))))


m <- stan_model(file = "stan/bayesian_nn.stan")
v <- vb(m, data = dat, init = list(W_raw = matrix(c(1/sqrt(2),-1/sqrt(2),1/sqrt(2),1/sqrt(2)),2), c_raw = c(0,5), w = c(1,-2)), grad_samples = 100, algorithm = "fullrank")

#fit in stan. Try different chains. Different inits etc.
dat <- list(N = N, X = cbind(synth$x1, synth$x2), y = synth$ynoise)
fit <- stan("bayesian_nn.stan", data = dat, chains = 1, iter = 2000, init = list(list(W_raw = matrix(c(1/sqrt(2),1/sqrt(2),1/sqrt(2),1/sqrt(2)),2), w = c(1,-2))))
fit <- stan("stan/bayesian_nn_angle.stan", data = dat, chains = 1, iter = 2000, init = list(list(angles = c(0.25,0.25), w = c(1,-2))))

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
  ggplot(aes(theta1,theta2)) + geom_raster(aes(fill = l)) +
  scale_fill_gradientn(colours = terrain.colors(10))

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
