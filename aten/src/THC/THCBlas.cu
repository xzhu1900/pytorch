#include <THC/THCBlas.h>
#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <algorithm>

float THCudaBlas_Sdot(THCState *state, int64_t n, float *x, int64_t incx, float *y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasSdot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Sdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

double THCudaBlas_Ddot(THCState *state, int64_t n, double *x, int64_t incx, double *y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    double result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasDdot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Ddot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

at::Half THCudaBlas_Hdot(THCState *state, int64_t n, at::Half *x, int64_t incx, at::Half *y, int64_t incy)
{
#if CUDA_VERSION >= 8000
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    at::Half result;
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(cublasDotEx(handle, n,
                              x, CUDA_R_16F, incx,
                              y, CUDA_R_16F, incy,
                              &result, CUDA_R_16F,
                              CUDA_R_32F));
    return result;
  }

  THError("Cublas_Hdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0.0;
#else
  THError("Cublas_Hdot requires CUDA 8.0+");
  return 0.0;
#endif
}

/* Level 2 */

void adjustLdLevel2(int64_t m, int64_t n, int64_t *lda)
{
  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  // TODO: why does Level3 check trans but this doesn't?
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

void THCudaBlas_Sgemv(THCState *state, char trans, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy)
{
  at::cuda::blas::gemv<float>(THCState_getCurrentStream(state), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void THCudaBlas_Dgemv(THCState *state, char trans, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy)
{
  at::cuda::blas::gemv<double>(THCState_getCurrentStream(state), trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void THCudaBlas_Sger(THCState *state, int64_t m, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda)
{
  adjustLdLevel2(m, n, &lda);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasSger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Sger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Dger(THCState *state, int64_t m, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda)
{
  adjustLdLevel2(m, n, &lda);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
      cublasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(cublasDger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Dger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}


cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    THError("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t *lda, int64_t *ldb, int64_t *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  if(n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if(transa_)
  {
    if(m <= 1)
      *lda = std::max<int64_t>(k, 1);
  }
  else
  {
    if(k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if(transb_)
  {
    if(k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  }
  else
  {
    if(n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }

}

/* Level 3 */
void THCudaBlas_Sgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  at::cuda::blas::gemm<float>(THCState_getCurrentStream(state), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// In CUDA 8.0, definition of data types for sgemmex changed
#if CUDA_VERSION < 8000
#  define CUDA_R_16F CUBLAS_DATA_HALF
#endif

void THCudaBlas_Hgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::Half alpha, at::Half *a, int64_t lda, at::Half *b, int64_t ldb, at::Half beta, at::Half *c, int64_t ldc)
{
  at::cuda::blas::gemm<at::Half>(THCState_getCurrentStream(state), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void THCudaBlas_Dgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  at::cuda::blas::gemm<double>(THCState_getCurrentStream(state), transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#if CUDA_VERSION >= 9010  || defined __HIP_PLATFORM_HCC__
void THCudaBlas_HgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             at::Half alpha, const at::Half *a, int64_t lda, int64_t strideA, const at::Half *b, int64_t ldb, int64_t strideB,
                             at::Half beta, at::Half *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )

  {
    THError("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  float fAlpha = alpha;
  float fBeta = beta;
#ifdef __HIP_PLATFORM_HCC__
  THCublasCheck(rocblas_gemm_strided_batched_ex(handle, opa, opb, (int)m, (int)n, (int)k,
                                   (void*)&fAlpha, a, rocblas_datatype_f16_r, (int)lda, strideA,
                                   b, rocblas_datatype_f16_r, (int)ldb, strideB,
                                   (void*)&fBeta, c, rocblas_datatype_f16_r, (int)ldc, strideC,
                                   c, rocblas_datatype_f16_r, (int)ldc, strideC,
                                   (int) batchCount, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                   0, 0, NULL, NULL));
#else
  THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  THCublasCheck(cublasGemmStridedBatchedEx(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   (void*)&fAlpha, a, CUDA_R_16F, (int)lda, strideA,
                                   b, CUDA_R_16F, (int)ldb, strideB,
                                   (void*)&fBeta, c, CUDA_R_16F, (int)ldc, strideC,
                                   (int)batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
}
#endif

void THCudaBlas_SgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
                             float beta, float *c[], int64_t ldc, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_SgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

#ifdef __HIP_PLATFORM_HCC__

  const int64_t stridea = (transa == 'N' || transa == 'n') ? lda*k : lda*n;
  const int64_t strideb = (transb == 'N' || transb == 'n') ? ldb*n : ldb*k;
  const int64_t stridec = ldc*n;

  THCudaBlas_SgemmStridedBatched(state, transa, transb, m, n, k, alpha, *a, lda, stridea, *b, ldb, strideb, beta, *c, ldc, stridec, batchCount);

#else

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasSgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
#endif
}

#if CUDA_VERSION >= 8000 || defined __HIP_PLATFORM_HCC__
void THCudaBlas_SgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             float alpha, const float *a, int64_t lda, int64_t strideA, const float *b, int64_t ldb, int64_t strideB,
                             float beta, float *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )

  {
    THError("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasSgemmStridedBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, strideA, b, (int)ldb, strideB, &beta, c, (int)ldc, strideC,
                                   (int)batchCount));
}
#endif

void THCudaBlas_DgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             double alpha, const double *a[], int64_t lda, const double *b[], int64_t ldb,
                             double beta, double *c[], int64_t ldc, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_DgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

#ifdef __HIP_PLATFORM_HCC__

  const int64_t stridea = (transa == 'N' || transa == 'n') ? lda*k : lda*n;
  const int64_t strideb = (transb == 'N' || transb == 'n') ? ldb*n : ldb*k;
  const int64_t stridec = ldc*n;

  THCudaBlas_DgemmStridedBatched(state, transa, transb, m, n, k, alpha, *a, lda, stridea, *b, ldb, strideb, beta, *c, ldc, stridec, batchCount);

#else

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasDgemmBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
#endif
}

#if CUDA_VERSION >= 8000 || defined __HIP_PLATFORM_HCC__
void THCudaBlas_DgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             double alpha, const double *a, int64_t lda, int64_t strideA, const double *b, int64_t ldb, int64_t strideB,
                             double beta, double *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_DgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasDgemmStridedBatched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, strideA, b, (int)ldb, strideB, &beta, c, (int)ldc, strideC,
                                   (int)batchCount));
}
#endif

void THCudaBlas_Sgetrs(THCState *state, char transa, int n, int nrhs, const float **a, int lda, int *pivot, float **b, int ldb, int *info, int batchSize)
{
#ifndef __HIP_PLATFORM_HCC__
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasSgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
#else
  THError("THCudaBlas_Sgetrs not supported in ROCM.");
#endif
}


void THCudaBlas_Dgetrs(THCState *state, char transa, int n, int nrhs, const double **a, int lda, int *pivot, double **b, int ldb, int *info, int batchSize)
{
#ifndef __HIP_PLATFORM_HCC__
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  cublasOperation_t opa = convertTransToCublasOperation(transa);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(cublasDgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
#else
  THError("THCudaBlas_Dgetrs not supported in ROCM.");
#endif
}

