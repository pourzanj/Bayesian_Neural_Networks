library(tidyverse)
library(rstan)
library(shinystan)
library(boot)
options(mc.cores = parallel::detectCores())

relu <- Vectorize(function(x) max(0,x))
log1p_exp <- Vectorize(function(x) log(1+exp(x)))

#create 1000 synthetic samples from log1p_exp network
N <- 100
synth <- tibble(x1 = runif(N,-10,10), x2 = runif(N,-10,10)) %>%
  mutate(h1 = relu((x1+x2)/sqrt(2)), h2 = relu((-x1+x2)/sqrt(2)-5)) %>%
  mutate(y = 1*h1-4*h2) %>%
  mutate(ynoise = y + rnorm(N, 0, 1))

synth %>% ggplot(aes(x1,x2, color = y)) + geom_point()
dat <- list(N = N, X = cbind(synth$x1, synth$x2), y = synth$ynoise)

fit <- stan("stan/bayesian_nn.stan", data = dat, chains = 1, iter = 2000, init = list(list(W_raw = matrix(c(1/sqrt(2),-1/sqrt(2),1/sqrt(2),1/sqrt(2)),2), c_raw = c(0,5), w = c(1,-2))))
fit_stiefel <- stan("stan/bayesian_nn_stiefel.stan", data = dat, chains = 1, iter = 2000, init = list(list(theta1_principal = as.array(pi/4), theta1_lower = as.array(0), theta2_principal = as.array(3*pi/4), theta2_lower = as.array(0), beta = c(1,-2), c_raw = c(0,5))))

pairs(fit, pars = c("W", "w", "c_raw", "beta1", "beta2"))
pairs(fit_stiefel, pars = c("theta1_principal", "theta2_principal", "beta", "c_raw"))

s$W_raw[,1,] %>% as_tibble %>% mutate(theta = atan2(V2,V1)) %>% mutate(theta_stiefel = extract(fit_stiefel)$theta1_principal[,1]) %>% select(-V1, -V2) %>% gather(var, value) %>% ggplot(aes(value)) + geom_histogram() + facet_grid(var ~ .)

fit <- stan("stan/bayesian_nn.stan", data = dat, chains = 1, iter = 2000, init = list(list(W_raw = matrix(rnorm(4),2), c_raw = sort(rnorm(2)), w = rnorm(2))))
fit <- stan("stan/bayesian_nn_stiefel.stan", data = dat, chains = 1, iter = 2000, init = list(list(theta1_principal = as.array(runif(1,-pi,pi)), theta1_lower = as.array(0), theta2_principal = as.array(runif(1,-pi,pi)), theta2_lower = as.array(0), beta = rnorm(2), c_raw = sort(rnorm(2)))))

pairs(fit_stiefel, pars = c("theta1_principal", "theta2_principal", "beta", "c_raw"))

m <- stan_model(file = "stan/bayesian_nn.stan")
v_mf <- vb(m, data = dat, init = list(W_raw = matrix(c(1/sqrt(2),-1/sqrt(2),1/sqrt(2),1/sqrt(2)),2), c_raw = c(0,5), w = c(1,-4)), grad_samples = 10, algorithm = "meanfield")
v_fr <- vb(m, data = dat, init = list(W_raw = matrix(c(1/sqrt(2),-1/sqrt(2),1/sqrt(2),1/sqrt(2)),2), c_raw = c(0,5), w = c(1,-4)), grad_samples = 10, algorithm = "fullrank")


m_stiefel <- stan_model(file = "stan/bayesian_nn_stiefel.stan")
v_stiefel <- vb(m_stiefel, data = dat, init = list(theta1_principal = as.array(pi/4), theta1_lower = as.array(0), theta2_principal = as.array(3*pi/4), theta2_lower = as.array(0), beta = c(1,-4), c_raw = c(0,5)), grad_samples = 100, algorithm = "meanfield")


names <- rownames(summary(v_stiefel)$summary)
names[1:4] <- c("theta1","theta2","beta1","beta2")
summary_vi <- summary(v)$summary %>% as_tibble %>% mutate(names = rownames(summary(v)$summary)) %>% select(-mean,-sd,-`25%`,-`75%`) %>% mutate(alg = "vi") %>% filter(names != "lp__") %>% filter(!str_detect(names,"raw"))
summary_stiefel_vi <- summary(v_stiefel)$summary %>% as_tibble %>% mutate(names = names) %>% select(-mean,-sd,-`25%`,-`75%`) %>% mutate(alg = "stiefel_vi") %>% filter(names != "lp__") %>% filter(!str_detect(names,"raw")) %>% filter(!str_detect(names,"w1")) %>% filter(!str_detect(names,"w2"))
summary_stiefel_hmc <- summary(fit_stiefel)$summary %>% as_tibble %>% mutate(names = names) %>% select(-mean,-sd,-`25%`,-`75%`,-n_eff,-Rhat,-se_mean) %>% mutate(alg = "stiefel_hmc") %>% filter(names != "lp__") %>% filter(!str_detect(names,"raw")) %>% filter(!str_detect(names,"w1")) %>% filter(!str_detect(names,"w2"))
summary_hmc <- summary(fit)$summary %>% as_tibble %>% mutate(names = rownames(summary(fit)$summary)) %>% select(-mean,-sd,-`25%`,-`75%`,-n_eff,-Rhat,-se_mean) %>% mutate(alg = "hmc") %>% filter(names != "lp__") %>% filter(!str_detect(names,"raw"))
rbind(summary_vi, summary_hmc) %>%
  rbind(summary_stiefel_vi) %>%
  rbind(summary_stiefel_hmc) %>%
  ggplot() +
  geom_point(aes(alg,`50%`)) +
  geom_errorbar(aes(x=alg,ymin=`2.5%`,ymax=`97.5%`)) +
  facet_wrap(~names,scales="free")



s <- extract(fit)
s_vi_mf <- extract(v_mf)
s_vi_fr <- extract(v_fr)

s_vi_stiefel <- extract(v_stiefel)

t <- tibble(W_vi = s$W_raw[,1,1], w_vi = s$w[,1]) %>% mutate(Algorithm = "True Posterior (HMC)")
t_vi_mf <- tibble(W_vi = s_vi_fr$W_raw[,1,1], w_vi = s_vi_fr$w[,1]) %>% mutate(Algorithm = "Mean-Field ADVI")
t_vi_fr <- tibble(W_vi = s_vi_mf$W_raw[,1,1], w_vi = s_vi_mf$w[,1]) %>% mutate(Algorithm = "Full-Rank ADVI")

rbind(t,t_vi_mf) %>% rbind(t_vi_fr) %>% mutate(Algorithm = factor(Algorithm)) %>%
ggplot(aes(W_vi, w_vi,color = Algorithm)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values=c("grey", "orange", "black")) + 
  xlab("W1[1,1]") + ylab("W2[1]") + theme_bw()

qplot(s$W_raw[,2,1],s$w[,2]) + geom_point(aes(W_vi, w_vi),data=tibble(W_vi = svi$W_raw[,2,1], w_vi = svi$w[,2]), color = "red")

qplot(s$beta2,s$w[,2]) + geom_point(aes(W_vi, w_vi),data=tibble(W_vi = svi$beta2, w_vi = svi$w[,2]), color = "red")


s_hmc_stiefel <- extract(fit_stiefel)
qplot(s_hmc_stiefel$beta[,1],s_hmc_stiefel$c_raw[,1]) + geom_point(aes(W_vi, w_vi),data=tibble(W_vi = s_vi_stiefel$beta[,1], w_vi = s_vi_stiefel$c_raw[,1]), color = "red")
qplot(s_hmc_stiefel$theta2_principal[,1],s_hmc_stiefel$beta[,2]) + geom_point(aes(W_vi, w_vi),data=tibble(W_vi = s_vi_stiefel$theta2_principal[,1], w_vi = s_vi_stiefel$beta[,2]), color = "red")


#givens transform plot
ggplot() +
  geom_density(aes(x), data = tibble(x = s$yhat[,1]), color = "black") +
  geom_histogram(aes(x,y=..density..), data = tibble(x = svi$yhat[,1]), fill = "orange", alpha = 0.4) +
  geom_histogram(aes(x,y=..density..), data = tibble(x = s_vi_stiefel$yhat[,1]), fill = "black", alpha = 0.4) +
  #geom_vline(xintercept = -36.56854, color = "black", size= 1) +
  xlab("y") +
  #xlim(-10,15)
  theme_bw()


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
