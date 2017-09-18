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
  matrix[N,2] X;
  vector[N] y;
}
parameters {
  vector[2] W_raw[2];
  //unit_vector[2] W_raw[2];
  //row_vector<lower=0>[2] c;
  //vector[2] w;
}
transformed parameters {
  matrix[2,2] W;
  W[,1] = W_raw[1];
  W[,2] = W_raw[2];
}
model {
  row_vector[2] c;
  matrix[N,2] h;
  vector[2] w = to_vector({1,-2});
  
  to_array_1d(W_raw[1]) ~ normal(0,1);
  to_array_1d(W_raw[2]) ~ normal(0,1);
  c = to_row_vector({0,5});
  h = log1p_exp(X*W - rep_matrix(c,N));
  //c[1] ~ normal(0,10);
  //c[2] ~ normal(-1,10);
  y ~ normal(h*w, 0.1);
}
generated quantities {
  //row_vector[2] c;
  //matrix[N,2] hhat;
  //c = to_row_vector({0,-1});
  //hhat = X*W + rep_matrix(c,N);
}
