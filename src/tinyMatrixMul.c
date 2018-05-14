#include <tinyMatrixMul.h>
#include <arm_neon.h>
#include <string.h>
#include "asmNeonApi.h"

#define ASM_TCOPY

void tcopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t N, uint32_t stride, float *pDst, uint32_t numThreads)
{
	uint32_t i = 0, j = 0;

	uint32_t KDiv4 = K>>2;
	uint32_t KHas2 = (K>>1)&1;
	uint32_t KHas1 = K&1;

	uint32_t NDiv4 = N>>2;
	uint32_t NHas2 = (N>>1)&1;
	uint32_t NHas1 = N&1;

	uint32_t *pSrcStart, *pDstStart;

	#pragma omp parallel for num_threads(numThreads) schedule(static)
	for(j = 0; j < NDiv4; j++)
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

	if(NHas2)
	{
		pSrcStart = (uint32_t *)pSrc + j*4*stride;
		pDstStart = (uint32_t *)pDst + j*4*K;
		
		#pragma omp parallel for num_threads(numThreads) schedule(static)
		for( i = 0; i < KDiv4; i++)
		{
			/* Do 4x2 patch copy */
#ifdef ASM_TCOPY
			tcopy_4x2_asm(pSrcStart + i*4, 4*stride, pDstStart + i*16);
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

	if (NHas1)
	{
		pSrcStart = (uint32_t *)pSrc + (N-1)*stride;
		pDstStart = (uint32_t *)pDst + (N-1)*K;
		memcpy(pDstStart, pSrcStart, K*sizeof(*pSrc));
	}
}

void ncopy_patch_4x4(const float *pSrc, uint32_t K, uint32_t M, uint32_t stride, float *pDst, uint32_t numThreads)
{
	uint32_t i = 0, j = 0;

	uint32_t KDiv4 = K>>2;
	uint32_t KHas2 = (K>>1)&1;
	uint32_t KHas1 = K&1;

	uint32_t MDiv4 = M>>2;
	uint32_t MHas2 = (M>>1)&1;
	uint32_t MHas1 = M&1;

	uint32_t *pSrcStart, *pDstStart;
	uint32_t *pDst2x4Start, *pDst1x4Start;

	pDst2x4Start = (uint32_t *)pDst + 4*K*MDiv4;
	pDst1x4Start = (uint32_t *)pDst2x4Start + 4*K*2;

	#pragma omp parallel for num_threads(numThreads) schedule(static)
	for(j = 0; j < KDiv4; j++)
	{
		pSrcStart = (uint32_t *)pSrc + j*4*stride;
		pDstStart = (uint32_t *)pDst + j*4*4;

		for( i = 0; i < MDiv4; i++)
		{
			/* Do 4x4 patch copy */
			ncopy_4x4_asm(pSrcStart + i*4, 4*stride, pDstStart + i*4*K);
		}

		if (MDiv4 > 0) pSrcStart += i*4;

		if(MHas2)
		{
			/* Do 2x4 patch copy */
			ncopy_2x4_asm(pSrcStart, 4*stride, pDst2x4Start + j*8);

			pSrcStart += 2;
		}

		if(MHas1)
		{
			/* Do 1x4patch copy */
			ncopy_1x4_asm(pSrcStart, 4*stride, pDst1x4Start + j*4);
		}
	}

	if(KHas2)
	{
		pSrcStart = (uint32_t *)pSrc + j*4*stride;
		pDstStart = (uint32_t *)pDst + j*4*4;

		#pragma omp parallel for num_threads(numThreads) schedule(static)
		for( i = 0; i < MDiv4; i++)
		{
			/* Do 4x2 patch copy */
			ncopy_4x2_asm(pSrcStart + i*4, 4*stride, pDstStart + i*4*K);
		}

		if (MDiv4 > 0) pSrcStart += i*4;

		if(MHas2)
		{
			uint32_t *pTmp = pDst2x4Start + j*8;

			/* Do 2x2 patch copy */
			*(pTmp+0) = *(pSrcStart);
			*(pTmp+1) = *(pSrcStart + stride);
			*(pTmp+2) = *(pSrcStart + 1);
			*(pTmp+3) = *(pSrcStart + stride + 1);

			pSrcStart += 2;
			pDst2x4Start += 4;
		}

		if(MHas1)
		{
			uint32_t *pTmp = pDst1x4Start + j*4;

			/* Do 1x2 patch copy */
			*(pTmp+0) = *(pSrcStart);
			*(pTmp+1) = *(pSrcStart + stride);

			pDst1x4Start += 2;
		}
	}

	if (KHas1)
	{
		pSrcStart = (uint32_t *)pSrc + (K-1)*stride;
		pDstStart = (uint32_t *)pDst + (K-1)*4;

		for( i = 0; i < MDiv4; i++)
		{
			/* Do 4x1 patch copy */
			ncopy_4x1_asm(pSrcStart + i*4, 4*stride, pDstStart + i*4*K);
		}
		
		if (MDiv4 > 0) pSrcStart += i*4;

		if(MHas2)
		{
			/* Do 2x1 patch copy */
			*(pDst2x4Start+0) = *(pSrcStart);
			*(pDst2x4Start+1) = *(pSrcStart + 1);

			pSrcStart += 2;
		}

		if(MHas1) *(pDst1x4Start+0) = *(pSrcStart);
	}
}

int tinyMatrixMul(const float *A, const float *B, const float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t numThreads)
{
	int ret = 0;

	printf("%s %d\n", __func__, __LINE__);
	return ret;
}
