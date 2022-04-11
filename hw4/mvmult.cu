// $ nvcc -arch=sm_61 mvmult.cu -o mvmult -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>

#include <omp.h>
#include <string>
#include <cmath>
#include <assert.h>



void mv_mult(double* c, const double* a, const double* A, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    //c[i] = 0.;
    for (long j = 0; j < N; j++) {
      c[i] += A[i*N+j] * a[j];
    }
  }
}

__global__
void mv_mult_kernel(double* c, const double* a, const double* A, long N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
      double temp = 0.;
      for (long j = 0; j < N; j++) {
        temp += A[i*N+j] * a[j];
      }
      c[i] = temp;
  }
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = (1UL<<14); // 2^25

  double* x = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));
  double* A = (double*) malloc(N*N * sizeof(double));

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  float ms;

  
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) {
    A[i] = 1.0/(i+1.);
  }
  for (long i = 0; i < N; i++) {
    x[i] = i+2.;
    z[i] = 0.;
    z_ref[i] = 0.;
  }
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

  double *A_d, *x_d, *z_d;
  cudaMalloc(&z_d, N*sizeof(double));
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&A_d, N*N*sizeof(double));
  
  double tt = omp_get_wtime();
  cudaMemcpy(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  printf("%25s", "Matrix-Vector");
  double ttinner = omp_get_wtime();
  cudaEventRecord(startEvent, 0);
  mv_mult_kernel<<<N/1024,1024>>>(z_d, x_d, A_d, N);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("%20.2f\n", N * 2 * N * sizeof(double) * 1e-6 / ms );
  double gpu_out = omp_get_wtime()-tt;
  double gpu_in = omp_get_wtime()-ttinner;
  //cudaDeviceSynchronize();
  double ttcpu = omp_get_wtime();
  mv_mult(z_ref, x, A, N);
  printf("Matrix-vector CPU %f s\n", omp_get_wtime()-ttcpu);
  cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  printf("Matrix-vector GPU %f s, %f s\n", gpu_out, gpu_in);

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Matrix-vector Error = %f\n", err);
  
  cudaFree(x_d);
  cudaFree(z_d);
  cudaFree(A_d);

  free(x);
  free(z);
  free(z_ref);
  free(A);

  return 0;
}

