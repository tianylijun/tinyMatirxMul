#include <sys/mman.h>
#include <tinyMatrixMul.h>
#include <arm_neon.h>
#include <string.h>
#include <arm_neon.h>
#include <sys/time.h>
#include "asmNeonApi.h"

//#define TIME_PROFILE_ENABLE
//#define NTCOPY_PRT_ENABLE
//#define USE_MALLOC

static inline void* tinyMalloc(uint32_t size)
{
#ifdef USE_MALLOC
	return malloc(size);
#else
	return mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
}

static inline int32_t tinyFree(void * pAddr, uint32_t size)
{
#ifdef USE_MALLOC
	free(pAddr);
#else
	return munmap(pAddr, size);
#endif
}

static int32_t tinySgemmUnit(const tinyMatrixCtx_S *pCtx, const float *A, const float *B, float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t threadId)
{
	int32_t ret = 0;
	uint32_t i, j, MDiv4, MHas2, MHas1, NDiv4, NHas2, NHas1, KDiv4, KHas2, KHas1, stride;
	const float *pWeight;
	float *pBCopy, *pOutCur;
	uint32_t curThreadId = 0;
#ifdef _OPENMP
	curThreadId = omp_get_thread_num();
#endif
#ifdef TIME_PROFILE_ENABLE
	struct timeval tv_s, tv_e;
#endif

#ifdef TIME_PROFILE_ENABLE
	gettimeofday(&tv_s, NULL);
#endif

	stride = pCtx->N;
	MDiv4 = M>>2; MHas2 = (M>>1)&1; MHas1 = M&1;
	NDiv4 = N>>2; NHas2 = (N>>1)&1; NHas1 = N&1;
	KDiv4 = K>>2; KHas2 = (K>>1)&1; KHas1 = K&1;

#if 0
	printf("[%d:%d][M N K Stride] [%d %d %d %d]\n"
		   "MDiv4: %03d, MHas2: %d MHas1: %d\n"
		   "NDiv4: %03d, NHas2: %d NNas1: %d\n"
		   "KDiv4: %03d, KHas2: %d KHas1: %d\n",
			curThreadId, threadId, M, N, K, stride,
			MDiv4, MHas2, MHas1, 
			NDiv4, NHas2, NHas1, 
			KDiv4, KHas2, KHas1);
#endif

	pBCopy = (float *)tinyMalloc(K*N*sizeof(float));
	if (NULL == pBCopy)
	{
		printf("ERR: [%s %d] No memory, %d\n", __func__, __LINE__, pCtx->APackSize);
		return -1;
	}

	ncopy_patch_4x4(B, K, N, stride, pBCopy, 1/*pCtx->numThreads*/);

#ifdef TIME_PROFILE_ENABLE
	gettimeofday(&tv_e, NULL);
	printf("[%d:%d]ncopyTime: %.1f\n", curThreadId, threadId, (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /1000.0f);
#endif

#ifdef NTCOPY_PRT_ENABLE
	{

		uint32_t i, j;
		printf("==================ncopyB================\n");
		for (i = 0; i < K; i++) {
			for (j = 0; j < N; j++) {
				printf("%.3f ", pBCopy[i * N + j]);
			}
			printf("\n");
		}
	}
#endif

#ifdef TIME_PROFILE_ENABLE
	gettimeofday(&tv_s, NULL);
#endif
	//#pragma omp parallel for num_threads(pCtx->numThreads)
	for (i = 0; i < MDiv4; i++)
	{
		pOutCur = C + i*stride*4;
		pWeight = A + i*4*K;

		//#pragma omp parallel for num_threads(pCtx->numThreads)
		for (j = 0; j < NDiv4; j++)
			tinySgemm4xkx4(pWeight, &pBCopy[j*K*4], &pOutCur[j*4], K, stride<<2, NULL);

		if (NHas2) tinySgemm4xkx2(pWeight, &pBCopy[NDiv4*K*4], pOutCur + NDiv4*4, K, stride<<2, NULL);
		if (NHas1) tinySgemm4xkx1(pWeight, &pBCopy[NDiv4*K*4 + NHas2*K*2], pOutCur + NDiv4*4 + NHas2*2, K, stride<<2, NULL);
	}

	if (MHas2)
	{
		float *pOutCur = C + MDiv4*stride*4;
		pWeight = A + MDiv4*4*K;

		for (j = 0; j < NDiv4; j++)
			tinySgemm2xkx4(pWeight, &pBCopy[j*K*4], &pOutCur[j*4], K, stride<<2, NULL);

		if (NHas2) tinySgemm2xkx2(pWeight, &pBCopy[NDiv4*K*4], pOutCur + NDiv4*4, K, stride<<2, NULL);
		if (NHas1) tinySgemm2xkx1(pWeight, &pBCopy[NDiv4*K*4 + NHas2*K*2], pOutCur + NDiv4*4 + NHas2*2, K, stride<<2, NULL);
	}

	if (MHas1)
	{
		float *pOutCur = C + (M-1)*stride;
		const float *pWeightCur = A + (M - 1)*K;

		for (j = 0; j < NDiv4; j++)
			tinySgemm1xkx4(pWeightCur, &pBCopy[j*K*4], &pOutCur[j*4], K, stride<<2, NULL);

		if (NHas2) tinySgemm1xkx2(pWeightCur, &pBCopy[NDiv4*K*4], pOutCur + NDiv4*4, K, stride<<2, NULL);
		if (NHas1) tinySgemm1xkx1(pWeightCur, &pBCopy[NDiv4*K*4 + NHas2*K*2], pOutCur + NDiv4*4 + NHas2*2, K, stride<<2, NULL);
	}

	tinyFree(pBCopy, K*N*sizeof(float));

#ifdef TIME_PROFILE_ENABLE
	gettimeofday(&tv_e, NULL);
	printf("[%d:%d]ComputeTime: %.1f\n", curThreadId, threadId, (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /1000.0f);
#endif

	return ret;
}

tinyMatrixCtx_S* tinyMatrixInit(const float *A, uint32_t M, uint32_t N, uint32_t K, uint32_t numThreads)
{
#ifdef TIME_PROFILE_ENABLE
	struct timeval tv_s, tv_e;
#endif
	tinyMatrixCtx_S *pCtx = malloc(sizeof(tinyMatrixCtx_S));
	if (NULL == pCtx)
	{
		printf("ERR: [%s %d] No memory, %lu\n", __func__, __LINE__, sizeof(tinyMatrixCtx_S));
		return NULL;
	}

	pCtx->APackSize = M*K*sizeof(*A);
	pCtx->pAPack = (float *)tinyMalloc(pCtx->APackSize);
	if (NULL == pCtx->pAPack)
	{
		printf("ERR: [%s %d] No memory, %d\n", __func__, __LINE__, pCtx->APackSize);
		free(pCtx);
		return NULL;
	}

	pCtx->M = M; pCtx->N = N; pCtx->K = K; pCtx->numThreads = numThreads;

#ifdef TIME_PROFILE_ENABLE
	gettimeofday(&tv_e, NULL);
#endif

	tcopy_patch_4x4(A, K, M, K, pCtx->pAPack, numThreads);

#ifdef TIME_PROFILE_ENABLE
	gettimeofday(&tv_e, NULL);
	printf("TcopyTime: %.1f\n", (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /1000.0f);
#endif

#ifdef NTCOPY_PRT_ENABLE
	{
		uint32_t i, j;
		printf("==================tcopyA================\n");
		for (i = 0; i < M; i++) {
			for (j = 0; j < K; j++) {
				printf("%.3f ", pCtx->pAPack[i * K + j]);
			}
			printf("\n");
		}
	}
#endif

	return pCtx;
}

int32_t tinyMatrixDeInit(tinyMatrixCtx_S *pCtx)
{
	if ((NULL != pCtx) && (NULL != pCtx->pAPack) && (0 != pCtx->APackSize))
	{
		tinyFree(pCtx->pAPack, pCtx->APackSize);
	}

	if (NULL != pCtx)
		free(pCtx);

	return 0;
}

int32_t tinyMatrixMul(tinyMatrixCtx_S *pCtx, const float *A, const float *B, float *C, uint32_t M, uint32_t N, uint32_t K, uint32_t numThreads)
{
    uint32_t tN = N / numThreads;
    tN = tN + (4 - tN % 4) % 4;
	//printf("tN: %d, Threads: %d, N: %d\n", tN, numThreads, N);

	if ( 1 == numThreads || N <= numThreads || N / numThreads < 4)
		return tinySgemmUnit(pCtx, A, B, C, M, N, K, 0);

	#pragma omp parallel for num_threads(numThreads)
	for (uint32_t i = 0; i < numThreads; ++i)
	{
		uint32_t cN = tN;
		if ((numThreads-1) == i) cN = N - i*tN;

		//printf("Thread: %d cN: %d\n", i, cN);
		tinySgemmUnit(pCtx, A, B + i*tN,  C + i*tN, M, cN, K, i);
	}

	return 0;
}