#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // Thread number
  int p = 8;
  // Allocate the final addition saving array, p+1 to avoid indexing error
  long* sums = (long*) malloc((p+1) * sizeof(long));
  // Size of array that each thread has
  int length = n / p;
  // Initialize arrays
  prefix_sum[0] = 0; 
  sums[0] = 0;

  #pragma omp parallel num_threads(p)
  {
    // Get thread ID
    int id = omp_get_thread_num();

    #pragma omp for schedule(static, length)
    for (long j = 1; j < n; j++) {
      // compute each local scan
      prefix_sum[j] = prefix_sum[j-1] + A[j-1];
    }
    // Store the ramp up constant in each scan
    sums[id+1] = prefix_sum[(id+1)*(length)];
  }

  // Compute the ramp up constant properly
  for (long i = 1; i < p; i++) {
    sums[i] += sums[i-1];
  }

  #pragma omp parallel num_threads(p) 
  {
    int id = omp_get_thread_num()+1;
    // Compute each element to the correct level with ramp up constant in "sums"
    #pragma omp for schedule(static, length)
    for (long j = 1; j < n; j++) {prefix_sum[j] += sums[id-1];}
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  /*
  for (long i = 0; i < N; i++) {
    printf("%ld, %ld, %ld\n", B0[i], B1[i], A[i]);
  }
  */

  long err = 0;
  for (long i = 0; i < N; i++) {
    err = std::max(err, std::abs(B0[i] - B1[i]));
  }
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
