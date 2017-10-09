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
  matrix[N,2] X;
  vector[N] y;
}

parameters {
  vector<lower = -pi(), upper = pi()>[1] theta1_principal;
  //vector<lower = -pi()/2, upper = pi()/2>[1] theta1_lower;
  
  vector<lower = -pi(), upper = pi()>[1] theta2_principal;
  //vector<lower = -pi()/2, upper = pi()/2>[1] theta2_lower;
  
  vector[2] beta;
  ordered[2] c_raw;
}

transformed parameters{
  //matrix[2, 1] w1;
  //matrix[2, 1] w2;
  vector[2] w1;
  vector[2] w2;
  
  matrix[2,2] W;
  row_vector[2] c;
  w1 = area_form_lp(theta1_principal, 2);
  w2 = area_form_lp(theta2_principal, 2);
  //w1 = area_form_lp(theta1_principal, theta1_lower, 2, 1);
  //w2 = area_form_lp(theta2_principal, theta2_lower, 2, 1);
  W = append_col(w1,w2);
  c = to_row_vector(c_raw);
}

model {
  matrix[N,2] h;
  h = relu(X*W - rep_matrix(c,N));
  
  y ~ normal(h*beta, 1.0);
}

generated quantities {
  vector[1] yhat;
  yhat = (relu(to_matrix(to_row_vector({-10,10})*W - c)))*beta;
}