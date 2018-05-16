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
#define VSRC_4S_A1     V1.4S
#define VSRC_4S_A2     V2.4S
#define VSRC_4S_A3     V3.4S

#define VSRC_2S_A0     V0.2S
#define VSRC_2S_A1     V1.2S
#define VSRC_2S_A2     V2.2S
#define VSRC_2S_A3     V3.2S

#define VSRC_4S_B0     V4.4S
#define VSRC_4S_B1     V5.4S
#define VSRC_4S_B2     V6.4S
#define VSRC_4S_B3     V7.4S

#define VSRC_4S_B0_S0  V4.S[0]
#define VSRC_4S_B0_L0  {V4.S}[0]
#define VSRC_4S_B0_S1  V4.S[1]
#define VSRC_4S_B0_S2  V4.S[2]
#define VSRC_4S_B0_S3  V4.S[3]

#define VSRC_2S_B0     V4.2S
#define VSRC_2S_B1     V5.2S
#define VSRC_2S_B2     V6.2S
#define VSRC_2S_B3     V7.2S

#define VSRC_S_A0      V0.S
#define VSRC_S_A1      V1.S
#define VSRC_S_A2      V2.S
#define VSRC_S_A3      V3.S

#define VSRC_4S_C0     V16.4S
#define VSRC_4S_C1     V17.4S
#define VSRC_4S_C2     V18.4S
#define VSRC_4S_C3     V19.4S

#define VSRC_4S_C0_0   {V16.S}[0]
#define VSRC_4S_C0_1   {V16.S}[1]
#define VSRC_4S_C0_2   {V16.S}[2]
#define VSRC_4S_C0_3   {V16.S}[3]

#define VSRC_2S_C0     V16.2S
#define VSRC_2S_C1     V17.2S
#define VSRC_2S_C2     V18.2S
#define VSRC_2S_C3     V19.2S

#define VSRC_16B_C0     V16.16B
#define VSRC_16B_C1     V17.16B
#define VSRC_16B_C2     V18.16B
#define VSRC_16B_C3     V19.16B

#define VSRC_8B_C0      V16.8B
#define VSRC_8B_C1      V17.8B
#define VSRC_8B_C2      V18.8B
#define VSRC_8B_C3      V19.8B

/* void fmul_4xkx1_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag) */

	.text
	.align 5
#ifdef __APPLE__
	.global _fmul_4xkx1_asm
_fmul_4xkx1_asm:
#else
	.global fmul_4xkx1_asm
fmul_4xkx1_asm:
#endif

	prfm PLDL1KEEP, [A, #64]
	sxtw CSTRIDEX, CSTRIDE
	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	lsr KDIV4, K, #2
	prfm PLDL1KEEP, [B, #64]
	and KHAS2, K, #2
	and KHAS1, K, #1

__LOOP_4_BEG:
	cmp KDIV4, #0
	beq __2_BEG

	/* transport first */
	ld4  {VSRC_4S_A0, VSRC_4S_A1, VSRC_4S_A2, VSRC_4S_A3}, [A], #64
    ld1  {VSRC_4S_B0}, [B], #16

    prfm PLDL1KEEP, [A, #64]

    fmla VSRC_4S_C0, VSRC_4S_A0, VSRC_4S_B0_S0
    fmla VSRC_4S_C0, VSRC_4S_A1, VSRC_4S_B0_S1
    prfm PLDL1KEEP, [B, #32]
    fmla VSRC_4S_C0, VSRC_4S_A2, VSRC_4S_B0_S2
    fmla VSRC_4S_C0, VSRC_4S_A3, VSRC_4S_B0_S3

	b __LOOP_4_BEG

__2_BEG:
	cmp KHAS2, #0
	beq __1_BEG

	/* transport first */
	ld2  {VSRC_4S_A0, VSRC_4S_A1}, [A], #32
    ld1  {VSRC_2S_B0}, [B], #8

    fmla VSRC_4S_C0, VSRC_4S_A0, VSRC_4S_B0_S0
    fmla VSRC_4S_C0, VSRC_4S_A1, VSRC_4S_B0_S1

__1_BEG:
	cmp KHAS1, #0
	beq __END

	ld1  {VSRC_4S_A0}, [A]
    ld1  VSRC_4S_B0_L0, [B]

    fmla VSRC_4S_C0, VSRC_4S_A0, VSRC_4S_B0_S0

__END:
	st1 VSRC_4S_C0_0, [C], CSTRIDEX
	st1 VSRC_4S_C0_1, [C], CSTRIDEX
	st1 VSRC_4S_C0_2, [C], CSTRIDEX
	st1 VSRC_4S_C0_3, [C]

	ret
#endif