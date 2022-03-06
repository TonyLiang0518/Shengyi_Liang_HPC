// g++ -fopenmp -O3 -march=native jacobi2d-omp.cpp -o jacobi2d-omp && ./jacobi2d-omp

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"


double* Laplace_Solve(long N) {
  // Initialize problem setup
  double h = 1./(N+1.);

  // Initialize discretization
  double* f = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* u = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* u_new = (double*) malloc((N+2) * (N+2) * sizeof(double));

  // Put initial values in vector and matrix
  for (int i = 0; i < (N+2) * (N+2); i++) {
    f[i] = 1.;
    u[i] = 0.;
    u_new[i] = 0.;
  }

  Timer t;
  t.tic();
  // Initialize variables to terminate iteration
  double diff;
  double result = 0.;
  // Compute initial residual
  # pragma omp parallel for reduction(+:result)
  for (int i = 1; i < N+1; i++) {
    for (int j = 1; j < N+1; j++) {
      diff = ( (1. / (h*h) * (-u[i-1+j*(N+2)]-u[i+(j-1)*(N+2)]-u[i+1+j*(N+2)]-u[i+(j+1)*(N+2)] + 4.*u[i+j*(N+2)])) - f[i+j*(N+2)] );
      result += diff * diff;
    }
  }
  result = sqrt(result);
  double init_resid = result;
  double resid = init_resid;
  int count = 0;
  double hsq = h*h;
  int i; int j;
  
  int num = 8;
  while (((init_resid / resid) < 1000000.) && (count <= 500000)) {
    
    // Jacobi updates on u only so there is no need to construct big matrix A, u is stored in manner u_ij = u[i+j*(N+2)] with boundary points included so coding is easy
    # pragma omp parallel for shared(u_new, u, f, hsq, N) private(i,j) num_threads(num)
    for (j = 1; j < N+1; j++) {
      for (i = 1; i < N+1 ; i++) {
        u_new[i+j*(N+2)] = 0.25 * (hsq*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)]);
      }
    }
    
    result = 0.;
    // Update to u while computing residual in one loop
    # pragma omp parallel for shared(u_new, u, N, f, hsq) private(i,j,diff) num_threads(num) reduction(+:result) 
    for (j = 1; j < N+1; j++) {
      for (i = 1; i < N+1 ; i++) {
        diff = ( (1. / (h*h) * (-u_new[i-1+j*(N+2)]-u_new[i+(j-1)*(N+2)]-u_new[i+1+j*(N+2)]-u_new[i+(j+1)*(N+2)] + 4.*u_new[i+j*(N+2)])) - f[i+j*(N+2)] );
        result += diff * diff;
        u[i+j*(N+2)] = u_new[i+j*(N+2)];
      }
    }

    resid = sqrt(result);
    count++;
    //printf("Initial Residual: %.9g, Round Residual: %.9g, Iterations: %10d\n", init_resid, resid, count);
  }
  free(u_new);
  double time = t.toc();
  printf("Initial Residual: %15f, Final Residual: %15f, Iterations: %10d, Time: %10f, Thread Number: %10d\n", init_resid, resid, count, time, num);
  free(f);
  return u;
}

int main(int argc, char** argv) {
  const long N = 100; 
  double* u = Laplace_Solve(N);
  // Print out resulting u for debugging

  free(u);
  return 0;
}