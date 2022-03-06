// g++ -fopenmp -O3 -march=native gs2d-omp.cpp -o gs2d-omp && ./gs2d-omp

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
  double* ur = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* ub = (double*) malloc((N+2) * (N+2) * sizeof(double));

  // Put initial values in vector and matrix
  for (int i = 0; i < (N+2) * (N+2); i++) {
    f[i] = 1.;
    u[i] = 0.;
    ur[i] = 0.;
  }

  
  Timer t;
  t.tic();
  // Initialize variable to terminate iteration
  double result = 0.;
  double diff = 0.;
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
  int num = 8;
  while (((init_resid / resid) < 1000000.) && (count <= 500000)) {

    // Gauss-Seidel updates on u only so there is no need to construct big matrix A, u is stored in manner u_ij = u[i+j*(N+2)] with boundary points included so coding is easy
    // Update red points
    #pragma omp parallel for num_threads(num)
    for (long j = 1; j < N+1; j+=2) {
      for (long i = 1; i < N+1 ; i+=2) {
        u[i+j*(N+2)] = 0.25 * (hsq*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)]);
      }
    }

    #pragma omp parallel for num_threads(num)
    for (long j = 2; j < N+1; j+=2) {
      for (long i = 2; i < N+1 ; i+=2) {
        u[i+j*(N+2)] = 0.25 * (hsq*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)]);
      }
    }

    //#pragma barrier


    // Update black points
    #pragma omp parallel for num_threads(num)
    for (long j = 1; j < N+1; j+=2) {
      for (long i = 2; i < N+1 ; i+=2) {
        u[i+j*(N+2)] = 0.25 * (hsq*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)]);
      }
    }
    #pragma omp parallel for num_threads(num)
    for (long j = 2; j < N+1; j+=2) {
      for (long i = 1; i < N+1 ; i+=2) {
        u[i+j*(N+2)] = 0.25 * (hsq*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)]);
      }
    }

    result = 0.;
    
    # pragma omp parallel for reduction(+:result) num_threads(num) //private(diff) 
    for (long j = 1; j < N+1; j++) {
      for (long i = 1; i < N+1; i++) {
        diff =  (1. / (hsq) * (-u[i-1+j*(N+2)]-u[i+(j-1)*(N+2)]-u[i+1+j*(N+2)]-u[i+(j+1)*(N+2)] + 4.*u[i+j*(N+2)])) - f[i+j*(N+2)] ;
        result += diff * diff;
        //result += fabs(u[i+j*(N+2)]);
      }
    }
    
    

    resid = sqrt(result);
    count++;
    //printf("Initial Residual: %.9g, Final Residual: %.9g, Iterations: %10d\n", init_resid, resid, count);
  }
  double time = t.toc();
  printf("Initial Residual: %15f, Final Residual: %15f, Iterations: %10d, Time: %10f\n", init_resid, resid, count, time);
  free(f);
  return u;
}

int main(int argc, char** argv) {
  const long N = 400; 
  double* u = Laplace_Solve(N);
  // Print out resulting u for debugging
  free(u);
  return 0;
}