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

#define VTMP_S3        s3

/* void fmul_1xkx1_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag) */

	.text
	.align 5
#ifdef __APPLE__
	.global _fmul_1xkx1_asm
_fmul_1xkx1_asm:
#else
	.global fmul_1xkx1_asm
fmul_1xkx1_asm:
#endif

	prfm PLDL1KEEP, [A, #64]
	sxtw CSTRIDEX, CSTRIDE
	lsr KDIV4, K, #2
	prfm PLDL1KEEP, [B, #64]
	and KHAS2, K, #2
	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	and KHAS1, K, #1

__LOOP_4_BEG:
	cbz KDIV4, __2_BEG

	/* transport first */
	ld1  {VSRC_4S_A0}, [A], #16
    ld1  {VSRC_4S_B0}, [B], #16

    prfm PLDL1KEEP, [A, #16]
	subs KDIV4, KDIV4, #1
    fmla VSRC_4S_C0, VSRC_4S_A0, VSRC_4S_B0
	prfm PLDL1KEEP, [B, #16]

	b __LOOP_4_BEG

__2_BEG:
	cbz KHAS2, __1_BEG

	/* sum in VSRC_2S_C0 */
	faddp VSRC_4S_C0, VSRC_4S_C0, VSRC_4S_C0

	/* transport first */
	ld1  {VSRC_2S_A0}, [A], #8
    ld1  {VSRC_2S_B0}, [B], #8

    fmla VSRC_2S_C0, VSRC_2S_A0, VSRC_2S_B0

__1_BEG:
	cbz KHAS1, __END

	/* sum in VSRC_2S_C0 */
	faddp VSRC_2S_C0, VSRC_2S_C0, VSRC_2S_C0

	ld1  VSRC_S_A0_L0, [A]
    ld1  VSRC_S_B0_L0, [B]

    fmul VTMP_S3, VSRC_S_A0, VSRC_S_B0

__END:
	fadd VSRC_S_C0, VSRC_S_C0, VTMP_S3 
	st1 VSRC_S_C0_L0, [C]

	ret
#endif