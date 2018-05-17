#ifndef __ASMNEONAPI_H
#define __ASMNEONAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t ncopy_4x4_asm_sparse(const uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
uint32_t ncopy_2x4_asm_sparse(const uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
uint32_t ncopy_1x4_asm_sparse(const uint32_t *pSrc, uint32_t stride, uint32_t *pDst);

void tcopy_4x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void tcopy_2x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void tcopy_1x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void tcopy_4x2_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void tcopy_4x1_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);

void ncopy_4x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_2x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_1x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_4x2_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_4x1_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);

void tinySgemm4xkx4(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void tinySgemm4xkx2(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void tinySgemm4xkx1(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);

void tinySgemm2xkx4(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void tinySgemm2xkx2(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void tinySgemm2xkx1(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);

void tinySgemm1xkx4(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void tinySgemm1xkx2(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void tinySgemm1xkx1(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);

#ifdef __cplusplus
}
#endif

#endif
