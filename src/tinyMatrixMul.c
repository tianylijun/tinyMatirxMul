#include <tinyMatrixMul.h>
#include <arm_neon.h>
#include <string.h>
#include "asmNeonApi.h"

void tcopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t M, uint32_t stride, float *pDst, uint32_t numThreads)
{
	uint32_t i = 0, j = 0;

	uint32_t KDiv4 = K>>2; uint32_t KHas2 = (K>>1)&1; uint32_t KHas1 = K&1;

	uint32_t MDiv4 = M>>2; uint32_t MHas2 = (M>>1)&1; uint32_t MHas1 = M&1;

	uint32_t *pSrcStart, *pDstStart;

#if 0
	printf("M: %d, k: %d\n", M, K);
	printf("KDiv4: %d, KHas2: %d KHas1: %d\n", KDiv4, KHas2, KHas1);
	printf("MDiv4: %d, MHas2: %d KMas1: %d\n", MDiv4, MHas2, MHas1);
#endif

	#pragma omp parallel for num_threads(numThreads) schedule(static)
	for(j = 0; j < MDiv4; j++)
	{
		pSrcStart = (uint32_t *)pSrc + j*4*stride;
		pDstStart = (uint32_t *)pDst + j*4*K;

		for( i = 0; i < KDiv4; i++)
		{
			/* Do 4x4 patch copy */
			tcopy_4x4_asm(pSrcStart + i*4, 4*stride, pDstStart + i*16);
		}

		if (KDiv4 > 0)
		{
			pSrcStart += i*4;
			pDstStart += i*16;
		}

		if(KHas2)
		{
			/* Do 2x4 patch copy */
			tcopy_2x4_asm(pSrcStart, 4*stride, pDstStart);

			pSrcStart += 2;
			pDstStart += 8;
		}

		if(KHas1)
		{
			/* Do 1x4 patch copy */
			tcopy_1x4_asm(pSrcStart, 4*stride, pDstStart);
		}
	}

	if(MHas2)
	{
		pSrcStart = (uint32_t *)pSrc + MDiv4*4*stride;
		pDstStart = (uint32_t *)pDst + MDiv4*4*K;
		
		#pragma omp parallel for num_threads(numThreads) schedule(static)
		for( i = 0; i < KDiv4; i++)
		{
			/* Do 4x2 patch copy */
			tcopy_4x2_asm(pSrcStart + i*4, 4*stride, pDstStart + i*8);
		}

		if (KDiv4 > 0)
		{
			pSrcStart += i*4;
			pDstStart += i*8;
		}

		if(KHas2)
		{
			/* Do 2x2 patch copy */
			*(pDstStart+0) = *(pSrcStart);
			*(pDstStart+1) = *(pSrcStart + 1);
			*(pDstStart+2) = *(pSrcStart + stride);
			*(pDstStart+3) = *(pSrcStart + stride + 1);

			pSrcStart += 2;
			pDstStart += 4;
		}

		if(KHas1)
		{
			/* Do 1x2 patch copy */
			*(pDstStart+0) = *(pSrcStart);
			*(pDstStart+1) = *(pSrcStart + stride);
		}
	}

	if (MHas1)
	{
		pSrcStart = (uint32_t *)pSrc + (M-1)*stride;
		pDstStart = (uint32_t *)pDst + (M-1)*K;
		memcpy(pDstStart, pSrcStart, K*sizeof(*pSrc));
	}
}

void ncopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t N, uint32_t stride, float *pDst, uint32_t numThreads)
{
	uint32_t i = 0, j = 0;

	uint32_t KDiv4 = K>>2;
	uint32_t KHas2 = (K>>1)&1;
	uint32_t KHas1 = K&1;

	uint32_t NDiv4 = N>>2;
	uint32_t NHas2 = (N>>1)&1;
	uint32_t NHas1 = N&1;

	uint32_t *pSrcStart, *pDstStart;
	uint32_t *pDst2x4Start, *pDst1x4Start;

	pDst1x4Start = pDst2x4Start = (uint32_t *)pDst + 4*K*NDiv4;
	if(NHas2) pDst1x4Start = (uint32_t *)pDst2x4Start + 2*K;

#if 0
	printf("K: %d, N: %d\n", K, N);
	printf("KDiv4: %d, KHas2: %d KHas1: %d\n", KDiv4, KHas2, KHas1);
	printf("NDiv4: %d, NHas2: %d NNas1: %d\n", NDiv4, NHas2, NHas1);
#endif

	#pragma omp parallel for num_threads(numThreads) schedule(static)
	for(j = 0; j < KDiv4; j++)
	{
		pSrcStart = (uint32_t *)pSrc + j*4*stride;
		pDstStart = (uint32_t *)pDst + j*4*4;

		for( i = 0; i < NDiv4; i++)
		{
			/* Do 4x4 patch copy */
			ncopy_4x4_asm(pSrcStart + i*4, 4*stride, pDstStart + i*4*K);
		}

		if (NDiv4 > 0) pSrcStart += i*4;

		if(NHas2)
		{
			/* Do 2x4 patch copy */
			ncopy_2x4_asm(pSrcStart, 4*stride, pDst2x4Start + j*8);

			pSrcStart += 2;
		}

		if(NHas1)
		{
			/* Do 1x4patch copy */
			ncopy_1x4_asm(pSrcStart, 4*stride, pDst1x4Start + j*4);
		}
	}

	if(KHas2)
	{
		pSrcStart = (uint32_t *)pSrc + KDiv4*4*stride;
		pDstStart = (uint32_t *)pDst + KDiv4*4*4;

		#pragma omp parallel for num_threads(numThreads) schedule(static)
		for( i = 0; i < NDiv4; i++)
		{
			/* Do 4x2 patch copy */
			ncopy_4x2_asm(pSrcStart + i*4, 4*stride, pDstStart + i*4*K);
		}

		if (NDiv4 > 0) pSrcStart += i*4;

		if(NHas2)
		{
			uint32_t *pTmp = pDst2x4Start + KDiv4*8;

			/* Do 2x2 patch copy */
			*(pTmp+0) = *(pSrcStart);
			*(pTmp+1) = *(pSrcStart + stride);
			*(pTmp+2) = *(pSrcStart + 1);
			*(pTmp+3) = *(pSrcStart + stride + 1);

			pSrcStart += 2;
		}

		if(NHas1)
		{
			uint32_t *pTmp = pDst1x4Start + KDiv4*4;

			/* Do 1x2 patch copy */
			*(pTmp+0) = *(pSrcStart);
			*(pTmp+1) = *(pSrcStart + stride);
		}
	}

	if (KHas1)
	{
		pSrcStart = (uint32_t *)pSrc + (K-1)*stride;
		pDstStart = (uint32_t *)pDst + 4*4*KDiv4;

		if (KHas2) pDstStart += 2*4; 

		for( i = 0; i < NDiv4; i++)
		{
			/* Do 4x1 patch copy */
			ncopy_4x1_asm(pSrcStart + i*4, 4*stride, pDstStart + i*4*K);
		}
		
		pSrcStart += NDiv4*4;

		if(NHas2)
		{
			uint32_t *pTmp = pDst2x4Start + KDiv4*8;
			if (KHas2) pTmp += 4;

			/* Do 2x1 patch copy */
			*(pTmp+0) = *(pSrcStart);
			*(pTmp+1) = *(pSrcStart + 1);

			pSrcStart += 2;
		}

		if(NHas1) 
		{
			*((uint32_t *)pDst + K*N - 1) = *(pSrcStart);
		}
	}
}

static void inline tinySgemm4xkx4(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{
	fmul_4xkx4_asm(A, B, C, K, CStride, pSparseFlag);
}

static void inline tinySgemm4xkx2(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{
	fmul_4xkx2_asm(A, B, C, K, CStride, pSparseFlag);
}

static void inline tinySgemm4xkx1(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{
	fmul_4xkx1_asm(A, B, C, K, CStride, pSparseFlag);
}

static void inline tinySgemm2xkx4(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{
	fmul_2xkx4_asm(A, B, C, K, CStride, pSparseFlag);
}

static void inline tinySgemm2xkx2(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{
	fmul_2xkx2_asm(A, B, C, K, CStride, pSparseFlag);
}

static void inline tinySgemm2xkx1(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{
	fmul_2xkx1_asm(A, B, C, K, CStride, pSparseFlag);
}

static void inline tinySgemm1xkx4(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{

}

static void inline tinySgemm1xkx2(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{

}

static void inline tinySgemm1xkx1(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag)
{

}

static int ncopy_4_k(const float *pSrc, float *pDst, uint32_t K, uint32_t stride, uint32_t numThreads, uint32_t *pSparseFlag)
{
	int ret = 0;
	uint32_t i, KDiv4, KHas2, KHas1;
	uint32_t stridex4 = 4*stride;

	KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

	for ( i = 0; i < KDiv4; i++)
	{
		pSparseFlag[i] = ncopy_4x4_asm_sparse((uint32_t *)pSrc, stridex4, (uint32_t *)pDst);

		pSrc += 4;
		pDst += 16;
	}

	if (KHas2)
	{
		pSparseFlag[KDiv4] = ncopy_2x4_asm_sparse((uint32_t *)pSrc, stridex4, (uint32_t *)pDst);

		pSrc += 2;
		pDst += 8;
	}

	if (KHas1)
		pSparseFlag[KDiv4+KHas2] = ncopy_1x4_asm_sparse((uint32_t *)pSrc, stridex4, (uint32_t *)pDst);

	return ret;
}

static int ncopy_2_k(const float *pSrc, float *pDst, uint32_t K, uint32_t stride, uint32_t numThreads)
{
	int ret = 0;
	uint32_t i, KDiv4, KHas2, KHas1;

	KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

	for ( i = 0; i < KDiv4; i++)
	{
		ncopy_4x2_asm((uint32_t *)pSrc, 4*stride, (uint32_t *)pDst);
		pSrc += 4;
		pDst += 8;
	}

	if (KHas2)
	{
		*(pDst+0) = *(pSrc);
		*(pDst+1) = *(pSrc + stride);
		*(pDst+2) = *(pSrc + 1);
		*(pDst+3) = *(pSrc + stride + 1);

		pSrc += 2;
		pDst += 4;
	}

	if (KHas1)
	{
		/* Do 1x2 patch copy */
		*(pDst+0) = *(pSrc);
		*(pDst+1) = *(pSrc + stride);
	}

	return ret;
}

static int tcopy_k_4(const float *pSrc, float *pDst, uint32_t K, uint32_t stride, uint32_t numThreads)
{
	int ret = 0;
	uint32_t i, KDiv4, KHas2, KHas1;
	uint32_t stridex4 = 4*stride;

	KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

	for ( i = 0; i < KDiv4; i++)
	{
		tcopy_4x4_asm((uint32_t *)pSrc, stridex4, (uint32_t *)pDst);

		pSrc += 4*stride;
		pDst += 16;
	}

	if (KHas2)
	{
		tcopy_4x2_asm((uint32_t *)pSrc, stridex4, (uint32_t *)pDst);

		pSrc += 2*stride;
		pDst += 8;
	}

	if (KHas1)
		tcopy_4x1_asm((uint32_t *)pSrc, stridex4, (uint32_t *)pDst);

	return ret;
}

static int tcopy_k_2(const float *pSrc, float *pDst, uint32_t K, uint32_t stride, uint32_t numThreads)
{
	int ret = 0;
	uint32_t i, KDiv4, KHas2, KHas1;

	KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

	for ( i = 0; i < KDiv4; i++)
	{
		tcopy_2x4_asm((uint32_t *)pSrc, 4*stride, (uint32_t *)pDst);

		pSrc += 4*stride;
		pDst += 8;
	}

	if (KHas2)
	{
		pDst[0] = *pSrc;
		pDst[1] = *(pSrc + 1);
		pDst[2] = *(pSrc + stride);
		pDst[3] = *(pSrc + stride + 1);

		pSrc += 2*stride;
		pDst += 4;
	}

	if (KHas1)
	{
		pDst[0] = pSrc[0];
		pDst[1] = pSrc[1];
	}

	return ret;
}

static int tcopy_k_1(const float *pSrc, float *pDst, uint32_t K, uint32_t stride, uint32_t numThreads)
{
	int ret = 0;
	uint32_t i, KDiv4, KHas2, KHas1;

	KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

	for ( i = 0; i < KDiv4; i++)
	{
		tcopy_1x4_asm((uint32_t *)pSrc, 4*stride, (uint32_t *)pDst);

		pSrc += 4*stride;
		pDst += 4;
	}

	if (KHas2)
	{
		pDst[0] = *pSrc;
		pDst[1] = *(pSrc + stride);

		pSrc += 2*stride;
		pDst += 2;
	}

	if (KHas1)
		pDst[0] = pSrc[0];

	return ret;
}

/* A[M, K] weight mat, B[K, N] img mat */
int tinyMatrixMul(const float *A, const float *B, float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t numThreads)
{
	#define MAX_THREAD_NUM (4)
	int ret = 0;
	uint32_t i, j, inputBufferIdx = 0;
	uint32_t MDiv4, MHas2, MHas1, NDiv4, NHas2, NHas1, KDiv4, KHas2, KHas1;
	float *pOut, *pWeight, *pInput[MAX_THREAD_NUM], *pOutCur;
	uint32_t *pSparseFlag[MAX_THREAD_NUM];

	MDiv4 = M>>2; MHas2 = (M>>1)&1; MHas1 = M&1;
	NDiv4 = N>>2; NHas2 = (N>>1)&1; NHas1 = N&1;
	KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

	if (numThreads > MAX_THREAD_NUM) numThreads = MAX_THREAD_NUM;

	/* float wwight[4*K] + float input[numThreads*4*K] + uint32_t SparseFlag[numThreads*(KDiv4 + KHas2 + KHas1)]*/
	pWeight = (float *)malloc((numThreads + 1)*4*K*sizeof(float) + numThreads*(KDiv4 + KHas2 + KHas1)*sizeof(uint32_t));
	if (NULL == pWeight)
	{
		printf("malloc failed, %lu\n", (numThreads + 1)*4*K*sizeof(A[0]));
		return -1;
	}

	for(i = 0; i < numThreads; i++)
	{
		pInput[i] = pWeight + (i + 1)*4*K;
		pSparseFlag[i] = (uint32_t *)(pWeight + (numThreads + 1)*4*K) + i*(KDiv4 + KHas2 + KHas1);
	}

	printf("[M N K - %d %d %d]\n", M, N, K);
	printf("MDiv4: %d, MHas2: %d KMas1: %d\n", MDiv4, MHas2, MHas1);
	printf("NDiv4: %d, NHas2: %d NNas1: %d\n", NDiv4, NHas2, NHas1);
	printf("KDiv4: %d, KHas2: %d KHas1: %d\n", KDiv4, KHas2, KHas1);

	//#pragma omp parallel for num_threads(numThreads)
	for (i = 0; i < MDiv4; i++)
	{
		pOutCur = C + i*N*4;

		#if 0//def OPENMP
		inputBufferIdx = omp_get_thread_num();
		#endif

		ncopy_4_k(A + i*4*K, pWeight, K, K, numThreads, pSparseFlag[inputBufferIdx]);

		for (j = 0; j < NDiv4; j++)
		{
			tcopy_k_4(B + j*4, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm4xkx4(pWeight, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);

			pOutCur += 4;
		}

		if (NHas2)
		{
			tcopy_k_2(B + NDiv4*4, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm4xkx2(pWeight, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);

			pOutCur += 2;
		}

		if (NHas1)
		{
			tcopy_k_1(B + N - 1, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm4xkx1(pWeight, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);
		}
	}

	if (MHas2)
	{
		float *pOutCur = C + MDiv4*N*4;
		ncopy_2_k(A + MDiv4*4*K, pWeight, K, K, numThreads);

		/* 4x2 */
		//#pragma omp parallel for num_threads(numThreads)
		for (j = 0; j < NDiv4; j++)
		{
			tcopy_k_4(B + j*4, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm2xkx4(pWeight, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);

			pOutCur += 4;
		}

		if (NHas2)
		{
			/* 2x2 */
			tcopy_k_2(B + NDiv4*4, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm2xkx2(pWeight, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);

			pOutCur += 2;
		}

		if (NHas1)
		{
			/* 1x2 */
			tcopy_k_1(B + N - 1, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm2xkx1(pWeight, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);
		}
	}

	if (MHas1)
	{
		float *pOutCur = C + (M-1)*N;
		const float *pWeightCur = A + (M - 1)*K;

		//#pragma omp parallel for num_threads(numThreads)
		for (j = 0; j < NDiv4; j++)
		{
			tcopy_k_4(B + j*4, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm1xkx4(pWeightCur, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);

			pOutCur += 4;
		}

		if (NHas2)
		{
			tcopy_k_2(B + NDiv4*4, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm1xkx2(pWeightCur, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);

			pOutCur += 2;
		}

		if (NHas1)
		{
			tcopy_k_1(B + N - 1, pInput[inputBufferIdx], K, N, numThreads);

			tinySgemm1xkx1(pWeightCur, pInput[inputBufferIdx], pOutCur, K, N, pSparseFlag[inputBufferIdx]);
		}
	}

	free(pWeight);
	return ret;
}
