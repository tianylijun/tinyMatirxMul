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

#ifdef __cplusplus
extern "C" {
#endif

void ncopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t N, uint32_t stride, float *pDst, uint32_t numThreads);
void tcopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t M, uint32_t stride, float *pDst, uint32_t numThreads);
int tinyMatrixMul(const float *A, const float *B, float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t num_threads);

#ifdef __cplusplus
}
#endif

#endif
