functions {

      matrix relu(matrix X) {

    int N;
    int M;
    matrix[dims(X)[1], dims(X)[2]] ret;
    N = dims(X)[1];
    M = dims(X)[2];

    for(i in 1:N) {
      for(j in 1:M) ret[i,j] = fmax(X[i,j], 0);
    }
    
    return ret;
  }
}

data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N,D] X;
  vector[N] y;
  
  int<lower=0> N_test;
  matrix[N_test,D] X_test;
}

parameters {
  matrix[D,50] W;
  
  vector[50] beta;
  row_vector[50] c;
  
  real<lower=0> sigma;
}

model {
  matrix[N,50] h;
  
  to_array_1d(W) ~ double_exponential(0, 1);
  beta ~ normal(0,1);
  c ~ normal(0,1);

  h = relu(X*W - rep_matrix(c,N));
  y ~ normal(h*beta, sigma);
}

generated quantities {
  vector[N_test] yhat;
  matrix[N_test,50] h;
  h = relu(X_test*W - rep_matrix(c,N_test));
  yhat = h*beta;
  for(n in 1:N_test) yhat[n] = normal_rng(yhat[n],sigma);
}
