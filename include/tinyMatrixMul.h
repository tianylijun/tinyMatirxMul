#ifndef __TINYMATRIXMUL_H
#define __TINYMATRIXMUL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tinyMatrixMul(const float *A, const float *B, const float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t num_threads);

#ifdef __cplusplus
}
#endif

#endif
