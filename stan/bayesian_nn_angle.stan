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
  real<lower=-1,upper=1> angles[2];
  //row_vector<lower=0>[2] c;
  //vector[2] w;
}
transformed parameters {
  matrix[2,2] W;
  W[1,1] = cos(angles[1]*pi());
  W[2,1] = sin(angles[1]*pi());
  W[1,2] = cos(angles[2]*pi());
  W[2,2] = sin(angles[2]*pi());
}
model {
  row_vector[2] c;
  matrix[N,2] h;
  vector[2] w = to_vector({1,-2});
  
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