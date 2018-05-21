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

int main(int argc, char* argv[]) {
	int i, j, loop;
	int m = 512;
	int n = 30;
	int k = 4608;
	int flag = 0;
	int loopcnt = 1;
	int fractions = 0;
	int numThreads = 0;
	int validCols = 4;
	struct timeval tv_s, tv_e;
	float total_time = .0f;
	long ttime = 0;

	printf("e.g. : ./sgemm-test-linux m n k loop threads print-flag\n");

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
		flag = atoi(argv[6]);
	if (argc > 7)
		validCols = atoi(argv[7]);

	printf("MNK [%d %d %d], print-flag: %d, loop: %d threads: %d, validCols: %d\n", m, n, k, flag, loopcnt, numThreads, validCols);
	
	float *A = (float *)malloc(2*(m*k + k*n)*sizeof(float));
    float *Aout = A + m*k;
    float *B = Aout + m*k;
    float *Bout = B + k*n;

    for (i = 0; i < m * k; i++) A[i] = (rand()%10)/10.0f;

    gettimeofday(&tv_s, NULL);

	for (i = 0 ; i < loopcnt; i++)
		tcopy_patch_4x4(A, k, m, k, Aout, numThreads);
	
	gettimeofday(&tv_e, NULL);

	if (0 != flag)
	{
		printf("==================A================\n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < k; j++) {
				printf("%.1f ", A[i * k + j]);
			}
			printf("\n");
		}
		printf("===============tcopyA==============\n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < k; j++) {
				printf("%.1f ", Aout[i * k + j]);
			}
			printf("\n");
		}
	}

	printf("tcopy avg time: %.1f\n", (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /(loopcnt* 1000.0f));

	for (i = 0; i < k * n; i++) B[i] = (rand()%10)/10.0f;

    gettimeofday(&tv_s, NULL);

	for (i = 0 ; i < loopcnt; i++)
		ncopy_patch_4x4(B, k, validCols, n, Bout, numThreads);
	
	gettimeofday(&tv_e, NULL);

	if (0 != flag)
	{
		printf("==================A================\n");
		for (i = 0; i < k; i++) {
			for (j = 0; j < n; j++) {
				printf("%.1f ", B[i * n + j]);
			}
			printf("\n");
		}
		printf("===============ncopyA==============\n");
		for (i = 0; i < k; i++) {
			for (j = 0; j < validCols; j++) {
				printf("%.1f ", Bout[i * validCols + j]);
			}
			printf("\n");
		}
	}
	printf("ncopy avg time: %.1f\n", (tv_e.tv_sec * 1000000 - tv_s.tv_sec * 1000000 + tv_e.tv_usec - tv_s.tv_usec) /(loopcnt* 1000.0f));
	
	free(A);
	return 0;
}
