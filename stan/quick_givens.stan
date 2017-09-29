functions {
  
  vector cumprod(vector x) {
    return(exp(cumulative_sum(log(x))));
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
    
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    cumprod_cos_theta = cumprod(reverse(cos_theta[1:(D-1)]));
    w = append_row(cumprod_cos_theta[D-1], sin_theta .* append_row(reverse(cumprod_cos_theta[1:(D-2)]), 1));
    
    v = append_row(1,rep_vector(0,D-1)) - w;
    H = diag_matrix(rep_vector(1,D));
    if(v'*v != 0) H = H +  2*(v*v')/(v'*v);
    
    J = append_row(-w[1]*(sin_theta ./ cos_theta)', diag_matrix(reverse(cumprod_cos_theta[1:(D-1)])));
    for(j in 2:(D-1)) J[j,j:(D-1)] = -w[j]*(sin_theta ./ cos_theta)'[j:(D-1)];
    
    target += log_determinant(H'[2:D,]*J);
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
  int D;
}

parameters {
  real<lower = -pi(), upper = pi()> theta_principal;
  vector<lower = -pi()/2, upper = pi()/2>[D-2] theta_lower;
}

transformed parameters{
  vector[D] w;
  w = area_form_lp(append_row(theta_principal, theta_lower), D);
}

model {
  
}
