// $ nvcc -arch=sm_61 mvmult.cu -o mvmult -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>

#include <omp.h>
#include <string>
#include <cmath>
#include <assert.h>


void vec_mult(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}





__global__
void vec_mult_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] * b[idx];
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = (1UL<<25); // 2^25

  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  float ms;

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2.;
    y[i] = 1.0/(i+1.);
    z[i] = 0.;
    z_ref[i] = 0.;
  }
  
  

  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, N*sizeof(double));

  double tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();

  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

  printf("%25s", "Inner product");
  cudaEventRecord(startEvent, 0);
  vec_mult_kernel<<<N/1024,1024>>>(z_d, x_d, y_d, N);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&ms, startEvent, stopEvent);
  printf("%20.2f\n", 2 * N * sizeof(double) * 1e-6 / ms );
  double gpu_out = omp_get_wtime()-tt;
  double gpu_in = omp_get_wtime()-ttinner;
  //cudaDeviceSynchronize();
  double ttcpu = omp_get_wtime();
  vec_mult(z_ref, x, y, N);
  printf("Inner product CPU %f s\n", omp_get_wtime()-ttcpu);

  cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
  printf("Inner product GPU %f s, %f s\n", gpu_out, gpu_in);

  double err = 0;
  for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  printf("Inner product Error = %f\n", err);
  
  
  
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);

  return 0;
}

