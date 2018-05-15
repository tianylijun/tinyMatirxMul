#include <tinyMatrixMul.h>
#include <arm_neon.h>
#include <string.h>
#include "asmNeonApi.h"

#define ASM_TCOPY

void tcopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t M, uint32_t stride, float *pDst, uint32_t numThreads)
{
	uint32_t i = 0, j = 0;

	uint32_t KDiv4 = K>>2;
	uint32_t KHas2 = (K>>1)&1;
	uint32_t KHas1 = K&1;

	uint32_t MDiv4 = M>>2;
	uint32_t MHas2 = (M>>1)&1;
	uint32_t MHas1 = M&1;

	uint32_t *pSrcStart, *pDstStart;

	//printf("M: %d, k: %d\n", M, K);
	//printf("KDiv4: %d, KHas2: %d KHas1: %d\n", KDiv4, KHas2, KHas1);
	//printf("MDiv4: %d, MHas2: %d KMas1: %d\n", MDiv4, MHas2, MHas1);

	#pragma omp parallel for num_threads(numThreads) schedule(static)
	for(j = 0; j < MDiv4; j++)
	{
		pSrcStart = (uint32_t *)pSrc + j*4*stride;
		pDstStart = (uint32_t *)pDst + j*4*K;

		for( i = 0; i < KDiv4; i++)
		{
			/* Do 4x4 patch copy */
#ifdef ASM_TCOPY
			tcopy_4x4_asm(pSrcStart + i*4, 4*stride, pDstStart + i*16);
#else
			uint32x4x4_t src32x4x4;
			src32x4x4.val[0] = vld1q_u32(pSrcStart + i*4 + 0*stride);
			src32x4x4.val[1] = vld1q_u32(pSrcStart + i*4 + 1*stride);
			vst1q_u32(pDstStart + i*16, src32x4x4.val[0]);
			src32x4x4.val[2] = vld1q_u32(pSrcStart + i*4 + 2*stride);
			vst1q_u32(pDstStart + i*16 + 4, src32x4x4.val[1]);
			src32x4x4.val[3] = vld1q_u32(pSrcStart + i*4 + 3*stride);
			vst1q_u32(pDstStart + i*16 + 8, src32x4x4.val[2]);
			vst1q_u32(pDstStart + i*16 + 12, src32x4x4.val[3]);
#endif
		}

		if (KDiv4 > 0)
		{
			pSrcStart += i*4;
			pDstStart += i*16;
		}

		if(KHas2)
		{
			/* Do 2x4 patch copy */
#ifdef ASM_TCOPY
			tcopy_2x4_asm(pSrcStart, 4*stride, pDstStart);
#else
			uint32x4_t src32x4;
			uint32x2x4_t src32x2x4;
			
			src32x2x4.val[0] = vld1_u32(pSrcStart + 0*stride);
			src32x2x4.val[1] = vld1_u32(pSrcStart + 1*stride);
			src32x4 = vcombine_u32(src32x2x4.val[0], src32x2x4.val[1]);
			vst1q_u32(pDstStart + 0, src32x4);
			
			src32x2x4.val[2] = vld1_u32(pSrcStart + 2*stride);
			src32x2x4.val[3] = vld1_u32(pSrcStart + 3*stride);
			src32x4 = vcombine_u32(src32x2x4.val[2], src32x2x4.val[3]);
			vst1q_u32(pDstStart + 4, src32x4);
#endif

			pSrcStart += 2;
			pDstStart += 8;
		}

		if(KHas1)
		{
			/* Do 1x4 patch copy */
#ifdef ASM_TCOPY
			tcopy_1x4_asm(pSrcStart, 4*stride, pDstStart);
#else
			uint32x4_t src32x4;
			vsetq_lane_u32(*(pSrcStart + 0*stride), src32x4, 0);
			vsetq_lane_u32(*(pSrcStart + 1*stride), src32x4, 1);
			vsetq_lane_u32(*(pSrcStart + 2*stride), src32x4, 2);
			vsetq_lane_u32(*(pSrcStart + 3*stride), src32x4, 3);

			vst1q_u32(pDstStart, src32x4);
#endif
		}
	}

#if 1
	if(MHas2)
	{
		pSrcStart = (uint32_t *)pSrc + MDiv4*4*stride;
		pDstStart = (uint32_t *)pDst + MDiv4*4*K;
		
		#pragma omp parallel for num_threads(numThreads) schedule(static)
		for( i = 0; i < KDiv4; i++)
		{
			/* Do 4x2 patch copy */
#ifdef ASM_TCOPY
			tcopy_4x2_asm(pSrcStart + i*4, 4*stride, pDstStart + i*8);
#else
			uint32x4x2_t src32x4x2;
			src32x4x2.val[0] = vld1q_u32(pSrcStart + i*4 + 0*stride);
			src32x4x2.val[1] = vld1q_u32(pSrcStart + i*4 + 1*stride);
			vst1q_u32(pDstStart + i*8, src32x4x2.val[0]);
			vst1q_u32(pDstStart + i*8 + 4, src32x4x2.val[1]);
#endif
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
#endif
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

	//printf("K: %d, N: %d\n", K, N);
	//printf("KDiv4: %d, KHas2: %d KHas1: %d\n", KDiv4, KHas2, KHas1);
	//printf("NDiv4: %d, NHas2: %d NNas1: %d\n", NDiv4, NHas2, NHas1);

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

int tinyMatrixMul(const float *A, const float *B, const float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t numThreads)
{
	int ret = 0;

	printf("%s %d\n", __func__, __LINE__);
	return ret;
}
