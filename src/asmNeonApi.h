#ifndef __ASMNEONAPI_H
#define __ASMNEONAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t tcopy_4x4_asm_sparse(const uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
uint32_t tcopy_2x4_asm_sparse(const uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
uint32_t tcopy_1x4_asm_sparse(const uint32_t *pSrc, uint32_t stride, uint32_t *pDst);

void tcopy_4x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void tcopy_2x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void tcopy_1x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void tcopy_4x2_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);

void ncopy_4x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_2x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_1x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_4x2_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);
void ncopy_4x1_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst);

#ifdef __cplusplus
}
#endif

#endif
