#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <tinyMatrixMul.h>


static void matricMul(int M, int N, int K, float*a, float*b, float *c) {
	int m, n, k;

	float maxA = 0.0f;
	float maxB = 0.0f;
	float maxC = 0.0f;
	float maxSum = 0.0f;
	float maxMulValue = 0.0f;

	float mul = 0.0f;

	for (m = 0; m < M; m++)
		for (n = 0; n < N; n++) {
			float sum = 0.0f;

			if (fabs(*(c + m * N + n)) > maxC)
				maxC = fabs(*(c + m * N + n));

			for (k = 0; k < K; k++) {
				if (fabs(*(a + m * K + k)) > maxA)
					maxA = fabs(*(a + m * K + k));
				if (fabs(*(b + k * N + n)) > maxB)
					maxB = fabs(*(b + k * N + n));

				mul = *(a + m * K + k) * *(b + k * N + n);

				if (fabs(mul) > maxMulValue)
					maxMulValue = fabs(mul);

				sum += mul;
			}

			if (fabs(sum) > maxSum)
				maxSum = fabs(sum);

			*(c + m * N + n) = sum;
		}

	printf("maxMul %f, maxA: %f, maxB:%f, maxC:%f, maxSum:%f\n",
					maxMulValue, maxA, maxB, maxC, maxSum);
}

int main(int argc, char* argv[]) {
	int i, j, loop, sameFlag;
	int m = 512;
	int n = 30;
	int k = 4608;
	int loopcnt = 1;
	int numThreads = 1;
	int prtFlag = 0;
	struct timeval tv_s, tv_e;

	printf("e.g. : %s m n k loop threads print-flag\n", argv[0]);

	if (argc > 1)
		m = atoi(argv[1]);
	if (argc > 2)
		n = atoi(argv[2]);
	if (argc > 3)
		k = atoi(argv[3]);
	if (argc > 4)
		loopcnt = atoi(argv[4]);
	if (argc > 5)
		numThreads = atoi(argv[5]);
	if (argc > 6)
		prtFlag = atoi(argv[6]);

	printf("MNK [%d %d %d], loop: %d threads: %d prtFlag: %d\n", m, n, k, loopcnt, numThreads, prtFlag);
	
	float *A = (float *)malloc((m*k + k*n + 4*m*n)*sizeof(float));
    float *B = A + m*k;
    float *C = B + k*n;
    float *C2 = C + m*n;
    float *C3 = C2 + m*n;
    float *CRef = C3 + m*n;

    for (i = 0; i < m * k; i++) A[i] = (rand()%10)/10.0f;
    for (i = 0; i < k * n; i++) B[i] = (rand()%10)/10.0f;
    
    if (prtFlag) printf("==================floatA================\n");
	for (i = 0; prtFlag && i < m; i++) {
		for (j = 0; j < k; j++) {
			printf("%.3f ", A[i * k + j]);
		}
		printf("\n");
	}
	if (prtFlag) printf("==================floatB================\n");
	for (i = 0; prtFlag && i < k; i++) {
		for (j = 0; j < n; j++) {
			printf("%.3f ", B[i * n + j]);
		}
		printf("\n");
	}
	if (prtFlag) printf("==================floatC================\n");
	matricMul(m,n,k,A,B,CRef);
	for (i = 0; prtFlag && i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%.3f ", CRef[i * n + j]);
		}
		printf("\n");
	}

	tinyMatrixCtx_S *pCtx = tinyMatrixInit(A, m, n, k, numThreads);
	if (NULL == pCtx)
	{
		free(A);
		return -1;
	}

    gettimeofday(&tv_s, NULL);

	for (i = 0 ; i < loopcnt; i++)
		tinyMatrixMul(pCtx, pCtx->pAPack, B, C, m, n, k, numThreads);
	
	gettimeofday(&tv_e, NULL);
	printf("tinysgemm avg time: %.1f\n", (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /(loopcnt* 1000.0f));

	tinyMatrixDeInit(pCtx);

	if (prtFlag) printf("==================tinyC================\n");
	for (i = 0; prtFlag && i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%.3f ", C[i * n + j]);
		}
		printf("\n");
	}

	sameFlag = 1;
	for (i = 0; i < m; i++) 
	{
		for (j = 0; j < n; j++)
		{
			if (fabs(*(C + i * n + j) - *(CRef + i * n + j))/fabs(*(C + i * n + j)) > 0.00001f)
			{
				sameFlag = 0;
				printf("asm diff with fix ref [%f != %f] diff [%f] [%x != %x] at (%d, %d)\n",
								*(C + i * n + j), *(CRef + i * n + j),
								fabs(*(C + i * n + j) - *(CRef + i * n + j)),
								*(unsigned int *)(C + i * n + j), *(unsigned int *)(CRef + i * n + j),
								i, j);
				break;
			}
		}
		if (0 == sameFlag)
			break;
	}

	printf("==================tinysgemm compare %s================\n\n\n",((1==sameFlag)?"same":"diff"));

	float *packA = (float *)malloc((m*k)*sizeof(float));
	externalPackA(m, k, packA, A, k);

	gettimeofday(&tv_s, NULL);
	
	for (i = 0 ; i < loopcnt; i++)
		block_sgemm_external_pack_threading(m, n, k, packA, B, C2, numThreads);

	gettimeofday(&tv_e, NULL);
	printf("blocksgemm avg time: %.1f\n", (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /(loopcnt* 1000.0f));

	sameFlag = 1;
	for (i = 0; i < m; i++) 
	{
		for (j = 0; j < n; j++)
		{
			if (fabs(*(C2 + i * n + j) - *(CRef + i * n + j))/fabs(*(C2 + i * n + j)) > 0.1f)
			{
				sameFlag = 0;
				printf("asm diff with fix ref [%f != %f] diff [%f] [%x != %x] at (%d, %d)\n",
								*(C2 + i * n + j), *(CRef + i * n + j),
								fabs(*(C2 + i * n + j) - *(CRef + i * n + j)),
								*(unsigned int *)(C2 + i * n + j), *(unsigned int *)(CRef + i * n + j),
								i, j);
				break;
			}
		}
		if (0 == sameFlag)
			break;
	}

	printf("==================block_sgemm4x4 compare %s================\n",((1==sameFlag)?"same":"diff"));

	gettimeofday(&tv_s, NULL);
	
	for (i = 0 ; i < loopcnt; i++)
		block_sgemm_external_pack_threading_8x8(m, n, k, packA, B, C3, numThreads);

	gettimeofday(&tv_e, NULL);
	printf("blocksgemm8x8 avg time: %.1f\n", (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /(loopcnt* 1000.0f));

	sameFlag = 1;
	for (i = 0; i < m; i++) 
	{
		for (j = 0; j < n; j++)
		{
			if (fabs(*(C3 + i * n + j) - *(CRef + i * n + j))/fabs(*(C3 + i * n + j)) > 0.1f)
			{
				sameFlag = 0;
				printf("asm diff with fix ref [%f != %f] diff [%f] [%x != %x] at (%d, %d)\n",
								*(C3 + i * n + j), *(CRef + i * n + j),
								fabs(*(C3 + i * n + j) - *(CRef + i * n + j)),
								*(unsigned int *)(C3 + i * n + j), *(unsigned int *)(CRef + i * n + j),
								i, j);
				break;
			}
		}
		if (0 == sameFlag)
			break;
	}

	printf("==================block_sgemm8x8 compare %s================\n",((1==sameFlag)?"same":"diff"));

	free(packA);
	free(A);
	return 0;
}
