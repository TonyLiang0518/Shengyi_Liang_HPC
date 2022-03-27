#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"
#include <cmath>

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// coefficients in the Taylor series expansion of sin(x) and cos(x)

static constexpr double c2  = -1/((double)2);
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
static constexpr double c12 =  1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
static constexpr double c13 =  1/(((double)2)*3*4*5*6*7*8*9*10*11*12*13);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11 + c13*x^13
// cos(x) = 1 - c2*x^2 + c4*x^4 + c6*x^6 + x8*x^8 + c10*x^10 + c12*x^12

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x_abs = abs(x[i]);
    double theta = fmod(x_abs, (M_PI*2.)) / 8;
    double signx = (x[i] > 0)-(x[i] < 0);

    //printf("%10f, %10f, %10d, %10f\n", x[i], theta, k, signx);

    double x1  = theta;
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x4  = x2 * x2;
    double x5  = x3 * x2;
    double x6  = x4 * x2;
    double x7  = x5 * x2;
    double x8  = x6 * x2;
    double x9  = x7 * x2;
    double x10 = x8 * x2;
    double x11 = x9 * x2;
    double x12 = x10 * x2;
    double x13 = x11 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    s += x13 * c13;

    double c = 1.;
    c += x2  * c2;
    c += x4  * c4;
    c += x6  * c6;
    c += x8  * c8;
    c += x10 * c10;
    c += x12 * c12;

    double s2 = s*s;
    double c2 = c*c;

    sinx[i] =  signx * 8. * s * c * (1. - 2.*s2) * (1. - 8. * s2 * c2);
    
  }
}



void sin4_intrin(double* sinx, const double* x, double* signx, double* theta, long N) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, _theta, _signx, _fifty_six, _eight;
  _signx  = _mm256_load_pd(signx);
  _fifty_six  = _mm256_set1_pd(56.);
  _eight  = _mm256_set1_pd(8.);

  x1 = _mm256_load_pd(theta);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x4  = _mm256_mul_pd(x2, x2);
  x5  = _mm256_mul_pd(x3, x2);
  x6  = _mm256_mul_pd(x4, x2);
  x7  = _mm256_mul_pd(x5, x2);
  x8  = _mm256_mul_pd(x6, x2);
  x9  = _mm256_mul_pd(x7, x2);
  x10 = _mm256_mul_pd(x8, x2);
  x11 = _mm256_mul_pd(x9, x2);
  x12 = _mm256_mul_pd(x10, x2);
  x13 = _mm256_mul_pd(x11, x2);

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x13 , _mm256_set1_pd(c13 )));

  __m256d c = _mm256_set1_pd(1.);
  c = _mm256_add_pd(c, _mm256_mul_pd(x2 , _mm256_set1_pd(c2 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x4 , _mm256_set1_pd(c4 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x6 , _mm256_set1_pd(c6 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x8 , _mm256_set1_pd(c8 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x10 , _mm256_set1_pd(c10 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x12 , _mm256_set1_pd(c12 )));

  __m256d s2 = _mm256_mul_pd(s, s);
  __m256d c2 = _mm256_mul_pd(c, c);
  __m256d term1 = _mm256_mul_pd(_mm256_set1_pd(8.), _mm256_mul_pd(s, c));
  __m256d term2 = _mm256_sub_pd(_mm256_set1_pd(1.), _mm256_mul_pd(_mm256_set1_pd(2.), s2));
  __m256d term3 = _mm256_sub_pd(_mm256_set1_pd(1.), _mm256_mul_pd(_mm256_set1_pd(8.), _mm256_mul_pd(s2, c2)));

  __m256d res = _mm256_mul_pd(term1, _mm256_mul_pd(term3, term2));
  res = _mm256_mul_pd(res, _signx);
  _mm256_store_pd(sinx, res);

#elif defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3, x5, x7, x9, x11;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);
    x5  = _mm_mul_pd(x3, x2);
    x7  = _mm_mul_pd(x5, x2);
    x9  = _mm_mul_pd(x7, x2);
    x11 = _mm_mul_pd(x9, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5 )));
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7 )));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9 )));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);

#endif

}

void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}
//double* zeros = (double*) aligned_malloc(N*sizeof(double));
int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));
  double* signx = (double*) aligned_malloc(N*sizeof(double));
  double* theta = (double*) aligned_malloc(N*sizeof(double));
  double x_abs;
  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI*8.; // [-pi/4,pi/4]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;

    x_abs = abs(x[i]);
    theta[i] = fmod(x_abs, (M_PI*2.)) / 8.;
    signx[i] = (x[i] > 0)-(x[i] < 0);
  }

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());
  
  tt.tic();
  
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));
  
  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i, signx+i, theta+i, N);
      
    }
  }
  //printf("%10f, %10f\n", sinx_taylor[500], x[500]);
  //printf("%10f, %10f\n", sinx_intrin[500], theta[500]);
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
  aligned_free(signx);
  aligned_free(theta);
}

