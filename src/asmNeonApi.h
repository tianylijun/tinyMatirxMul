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

void fmul_4xkx4_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void fmul_4xkx2_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void fmul_4xkx1_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);

void fmul_2xkx4_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void fmul_2xkx2_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void fmul_2xkx1_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);

void fmul_1xkx4_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void fmul_1xkx2_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);
void fmul_1xkx1_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag);

#ifdef __cplusplus
}
#endif

#endif
