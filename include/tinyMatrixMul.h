#ifndef __TINYMATRIXMUL_H
#define __TINYMATRIXMUL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MAX_THREAD_NUM
#define MAX_THREAD_NUM (4)
#endif

#define PAGE_SIZE_4K (4096)
#define SIZE_ALIGN(size, alignment) ((size + alignment - 1) & (~alignment))

typedef struct __tinyMatrixCtx
{
	float *pAPack;
	uint32_t APackSize;
	uint32_t *pSparseFlag;
	uint32_t M;
	uint32_t N;
	uint32_t K;
}tinyMatrixCtx_S;

#ifdef __cplusplus
extern "C" {
#endif

void ncopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t N, uint32_t stride, float *pDst, uint32_t numThreads);
void tcopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t M, uint32_t stride, float *pDst, uint32_t numThreads);

tinyMatrixCtx_S* tinyMatrixInit(const float *A, uint32_t M, uint32_t N, uint32_t K ,uint32_t numThreads);
int32_t tinyMatrixMul(tinyMatrixCtx_S *pCtx, const float *A, const float *B, float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t numThreads);
int32_t tinyMatrixDeInit(tinyMatrixCtx_S *pCtx);

#if 1
void externalPackA(int M, int L, float* packA, float* a, int lda);//External packing for A, requires space allocation for packA
void block_sgemm_external_pack_threading( int M, int N, int L, float *A, float *B, float *C, int num_threads);

void externalPackA8(int M, int L, float* packA, float* a, int lda);//External packing for A, requires space allocation for packA
void block_sgemm_external_pack_threading_8x8( int M, int N, int L, float *A, float *B, float *C, int num_threads);
#endif

#ifdef __cplusplus
}
#endif

#endif
