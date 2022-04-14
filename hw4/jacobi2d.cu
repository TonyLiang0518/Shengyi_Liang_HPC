// $ nvcc -arch=sm_61 jacobi2d.cu -o jacobi2d -Xcompiler -fopenmp
#include <string>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"
#include <assert.h>


double resid_comp(double* u, double* f, long N, double h) {
  double diff;
  double result = 0.;

  # pragma omp parallel for reduction(+:result)
  for (int i = 1; i < N+1; i++) {
    for (int j = 1; j < N+1; j++) {
      diff = ( (1. / (h*h) * (-u[j-1+i*(N+2)]-u[j+(i-1)*(N+2)]-u[j+1+i*(N+2)]-u[j+(i+1)*(N+2)] + 4.*u[j+i*(N+2)])) - f[j+i*(N+2)] );
      result += diff * diff;
    }
  }
  return sqrt(result);
}

__global__ 
void jacobi_kernel(double* u_new, double* u, double* f, long N, double hsq) {
  //int i = blockIdx.x * blockDim.x + threadIdx.x;
  //int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / (N+2);
  int j = idx - i*(N+2);
  double up, down, left, right, f_val, hsq_d;
  if (i > 0 && i < N+1 && j > 0 && j < N+1) {
    //printf("%10f, ",u_new[j+i*(N+2)]);
    up = u[j+(i-1)*(N+2)];
    down = u[j+(i+1)*(N+2)];
    left = u[j-1+i*(N+2)];
    right = u[j+1+i*(N+2)];
    f_val = f[j+i*(N+2)];
    hsq_d = hsq;
    
    double temp = 0.25 * (hsq_d * f_val + up + down + left + right);
    //printf("%10d, %10d, %10f\n", i,j,temp);
    u_new[j+i*(N+2)] = temp;
    //printf("%10d, %10d, %10f, %10f, %10f, %10f, %10f, %10f\n", i, j, up, down, left, right, temp, u_new[j+i*(N+2)]);
    //printf("%10d, %10d, %10f\n", i,j,u_new[j+i*(N+2)]);
  }
}

double* Laplace_Solve_Cuda(long N){
  // Initialize problem setup
  double h = 1./(N+1.);

  // Initialize discretization
  double* f = (double*) malloc((N+2) * (N+2) * sizeof(double));
  double* u = (double*) malloc((N+2) * (N+2) * sizeof(double));

  // Put initial values in vector and matrix
  for (int i = 0; i < (N+2) * (N+2); i++) {
    f[i] = 1.;
    u[i] = 0.;
  }

  double *f_d, *u_d, *u_new_d;
  cudaMalloc(&f_d, (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&u_d, (N+2)*(N+2)*sizeof(double));
  cudaMalloc(&u_new_d, (N+2)*(N+2)*sizeof(double));
  cudaMemcpy(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(u_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);

  Timer t;
  t.tic();
  // Initialize variables to terminate iteration
  double resid;
  double init_resid = resid_comp(u,f,N,h);
  double hsq = h*h;
  int count = 0;
  //dim3 blocks(N+2,N+2);
  //dim3 threads(1,1);
  while (count < 30000) {
    
    jacobi_kernel<<<N+2,N+2>>>(u_new_d, u_d, f_d, N, hsq);
    
    cudaMemcpyAsync(u_d, u_new_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    //printf("%s\n", cudaGetErrorString(cudaMemcpy(u_d, u_new_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToDevice)));
    count++;
  }
  //cudaMemcpy(f, f_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
  //printf("u: %10f\n", u[3*(N+2)+3]);
  //printf("%s\n", cudaGetErrorString(cudaMemcpy(u, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost)));
  cudaMemcpy(u, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
  double time = t.toc();
  //printf("u: %10f\n", u[3*(N+2)+3]);
  resid = resid_comp(u, f, N, h);
  printf("GPU: Initial Residual: %15f, Final Residual: %15f, Iterations: %10d, Time: %10f\n", init_resid, resid, count, time);
  free(f);

  cudaFree(u_d);
  cudaFree(f_d);
  cudaFree(u_new_d);
  return u;
}



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
  double resid;
  double init_resid = resid_comp(u,f,N,h);
  int count = 0;
  double hsq = h*h;
  int i; int j;
  
  int num = 8;
  while (count < 30000) {
    
    // Jacobi updates on u only so there is no need to construct big matrix A, u is stored in manner u_ij = u[i+j*(N+2)] with boundary points included so coding is easy
    # pragma omp parallel for shared(u_new, u, f, hsq, N) private(i,j) num_threads(num)
    for (j = 1; j < N+1; j++) {
      for (i = 1; i < N+1 ; i++) {
        u_new[j+i*(N+2)] = 0.25 * (hsq*f[j+i*(N+2)] + u[j-1+i*(N+2)] + u[j+(i-1)*(N+2)] + u[j+1+i*(N+2)] + u[j+(i+1)*(N+2)]);
      }
    }
    
    // Update to u while computing residual in one loop
    # pragma omp parallel for shared(u_new, u, N, f, hsq) private(i,j) num_threads(num)
    for (j = 1; j < N+1; j++) {
      for (i = 1; i < N+1 ; i++) {
        u[j+i*(N+2)] = u_new[j+i*(N+2)];
      }
    }


    count++;
    //printf("Initial Residual: %.9g, Round Residual: %.9g, Iterations: %10d\n", init_resid, resid, count);
  }
  resid = resid_comp(u, f, N, h);
  free(u_new);
  double time = t.toc();
  printf("CPU: Initial Residual: %15f, Final Residual: %15f, Iterations: %10d, Time: %10f, Thread Number: %10d\n", init_resid, resid, count, time, num);
  free(f);
  return u;
}

int main(int argc, char** argv) {
  const long N = 100; 
  double* u = Laplace_Solve(N);
  // Print out resulting u for debugging
  double* u2 = Laplace_Solve_Cuda(N);

  double err = 0;
  for (long i = 0; i < (N+2)*(N+2); i++) err += fabs(u[i]-u2[i]);
  printf("Error = %f\n", err);

  free(u);
  return 0;
}