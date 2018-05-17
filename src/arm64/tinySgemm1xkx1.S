#ifdef __aarch64__
/* RSV X19~X28 */
/**************in param**************/
#define A 		       x0
#define B		       x1
#define C 		       x2
#define K              w3
#define CSTRIDE        w4
#define CSTRIDEX       x4
#define SPARSE         x5

#define KDIV4          w6
#define KHAS2          w7
#define KHAS1          w8

/* RSV V8~V15 */
#define VSRC_4S_A0     V0.4S
#define VSRC_S_A0      s0
#define VSRC_S_A0_L0   {v0.s}[0]

#define VSRC_4S_B0     V1.4S
#define VSRC_S_B0      s1
#define VSRC_S_B0_L0   {v1.s}[0]

#define VSRC_2S_A0     V0.2S
#define VSRC_2S_B0     V1.2S

#define VSRC_4S_C0     V2.4S
#define VSRC_2S_C0     V2.2S
#define VSRC_S_C0      s2
#define VSRC_S_C0_L0   {V2.S}[0]
#define VSRC_16B_C0    V2.16B

#define VSRC_2S_C1     V3.2S
#define VSRC_8B_C1     V3.8B
#define VTMP_S3        s3
/* void tinySgemm1xkx1(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag) */

	.text
	.align 5
#ifdef __APPLE__
	.global _tinySgemm1xkx1
_tinySgemm1xkx1:
#else
	.global tinySgemm1xkx1
tinySgemm1xkx1:
#endif

	prfm PLDL1KEEP, [A, #64]
	lsr KDIV4, K, #2
	prfm PLDL1KEEP, [B, #64]
	and KHAS2, K, #2
	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	and KHAS1, K, #1
	eor VSRC_8B_C1, VSRC_8B_C1, VSRC_8B_C1

__LOOP_4_BEG:
	cbz KDIV4, __2_BEG

	ld1  {VSRC_4S_A0}, [A], #16
    ld1  {VSRC_4S_B0}, [B], #16

    prfm PLDL1KEEP, [A, #16]
    subs KDIV4, KDIV4, #1
    prfm PLDL1KEEP, [B, #16]
    fmla VSRC_4S_C0, VSRC_4S_A0, VSRC_4S_B0

	b __LOOP_4_BEG

__2_BEG:
	cbz KHAS2, __1_BEG

	/* transport first */
	ld1  {VSRC_2S_A0}, [A], #8
    ld1  {VSRC_2S_B0}, [B], #8

    fmul VSRC_2S_C1, VSRC_2S_A0, VSRC_2S_B0
    /* sum in s3 VTMP_S3 */
	faddp VSRC_2S_C1, VSRC_2S_C1, VSRC_2S_C1

__1_BEG:
	cbz KHAS1, __END

	ld1  VSRC_S_A0_L0, [A]
    ld1  VSRC_S_B0_L0, [B]
    /* sum in s0 */
    fmul VSRC_S_A0, VSRC_S_A0, VSRC_S_B0

__END:
	/* pack 4 --> 2 */
	faddp VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0
	/* sum in s2 VSRC_S_C0 pack 2 --> 1 */
	faddp VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0

	/* s2 = s2 + s3 */
	fadd VSRC_S_C0, VSRC_S_C0, VTMP_S3
	/* s2 = s2 + s0 */
	fadd VSRC_S_C0, VSRC_S_C0, VSRC_S_A0

	st1 VSRC_S_C0_L0, [C]

	ret
#endif