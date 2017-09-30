library(tidyverse)
library(rstan)

area_form <- function(theta) {
  
  D <- length(theta) + 1
  sin_theta = sin(theta)
  cos_theta = cos(theta)
  cumprod_cos_theta = cumprod(cos_theta[(D-1):1])
  w = c(cumprod_cos_theta[D-1], sin_theta * c(cumprod_cos_theta[(D-2):1],1))
  
  v = c(1,0,0) - w
  if((t(v) %*% v)[1,1] != 0) { H = diag(D) - 2*(v %*% t(v))/(t(v) %*% v)[1,1]}
  else {H = diag(D)}
    
  J = rbind(-w[1]*(sin_theta/cos_theta), diag(cumprod_cos_theta[(D-1):1]))
  for(j in 2:(D-1)) J[j,j:(D-1)] = -(sin_theta/cos_theta)[j:(D-1)]*w[j]

  abs(det((t(H)[2:D,]) %*% J))
}

fit <- stan(file = "stan/quick_givens.stan", data = list(D = 3), chains = 1, iter = 20)
fit_old <- stan(file = "stan/test_stiefel.stan", data = list(n = 3), chains = 1, iter = 20000)

s <- extract(fit)
s$theta_principal %>% qplot
s$theta_lower[,1] %>% qplot

S <- 10000
rand_unit <- map(1:S, function(s){ w <- rnorm(3); return(w/sqrt(sum(w^2)))}) %>%
  map(function(w) GivensTransform(as.matrix(w))) %>%
  map(function(w) matrix(w,nrow=1) %>% as_tibble) %>%
  bind_rows()

rand_unit %>% ggplot(aes(V1)) + geom_histogram()
rand_unit %>% ggplot(aes(V2)) + geom_histogram()
