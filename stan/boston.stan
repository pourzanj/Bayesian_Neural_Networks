functions {
  
     vector cumprod(vector x) {
    //return(exp(cumulative_sum(log(x))));
    vector[num_elements(x)] ret;
    int D = num_elements(x);
    ret[1] = x[1];
    for(i in 2:D) ret[i] = x[i] * ret[i-1];
    
    return(ret);
  }
  
  vector reverse(vector x) {
    vector[num_elements(x)] r;
    int D = num_elements(x);
    for(i in 1:D) r[i] = x[D-i+1];
    
    return(r);
  }
  
  vector area_form_lp(vector theta, int D) {
    
    vector[D] w;
    vector[D-1] sin_theta;
    vector[D-1] cos_theta;
    vector[D-1] cumprod_cos_theta;
    vector[D] v;
    matrix[D,D] H;
    matrix[D,D-1] J;
    matrix[D-1,D-1] area_forms;
    
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    cumprod_cos_theta = cumprod(reverse(cos_theta[1:(D-1)]));
    w = append_row(cumprod_cos_theta[D-1], sin_theta .* append_row(reverse(cumprod_cos_theta[1:(D-2)]), 1));
    
    v = append_row(1,rep_vector(0,D-1)) - w;
    H = diag_matrix(rep_vector(1,D));
    if(v'*v != 0) H = H - 2*(v*v')/(v'*v);
    
    J = append_row(-w[1]*(sin_theta ./ cos_theta)', diag_matrix(reverse(cumprod_cos_theta[1:(D-1)])));
    for(j in 2:(D-1)) J[j,j:(D-1)] = -w[j]*(sin_theta ./ cos_theta)'[j:(D-1)];
    
    //if(fabs(theta[2]) < 0.01) {
      //print("~~~~~",theta[1],"~~~~~");
      //print("cos_theta", cos_theta);
      //print("cumprod_cos_theta", cumprod_cos_theta);
      //print("w",w);
      //print("H",H);
      //print("J",J);
      //print(H'[2:D,]*J);
      //print("log_det", log_determinant(H'[2:D,]*J));
    //}
    
    area_forms = H'[2:D,]*J;
    //target += log_determinant(area_forms);
    return(w);
  }
    
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
  real<lower = -pi(), upper = pi()> theta_principal[50];
  vector<lower = -pi()/2, upper = pi()/2>[D-2] theta_lower[50];
  
  vector[50] beta;
  ordered[50] c_raw;
  
  real<lower=0> sigma;
  
  // real<lower=0> theta_p_h;
  // real<lower=0> theta_l_h;
  // real<lower=0> beta_h;
  // real<lower=0> c_h;
}

transformed parameters{
  matrix[D,50] W;
  row_vector[50] c;
  for(i in 1:50) {
    W[,i] = area_form_lp(append_row(theta_principal[i], theta_lower[i]), D);
  }
  c = to_row_vector(c_raw);
}

model {
  matrix[N,50] h;
  
  for(i in 1:50) theta_principal[i] ~ double_exponential(0, 1);
  for(i in 1:50) theta_lower[i] ~ double_exponential(0, 1);
  beta ~ normal(0,1);
  c_raw ~ normal(0,1);
  
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
