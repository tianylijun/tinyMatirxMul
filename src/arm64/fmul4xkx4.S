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

.macro fmla_col_4x4x4 sum_q, col0
     fmla    \sum_q, VSRC_4S_B0, \col0[0]
     fmla    \sum_q, VSRC_4S_B1, \col0[1]
     fmla    \sum_q, VSRC_4S_B2, \col0[2]
     fmla    \sum_q, VSRC_4S_B3, \col0[3]
.endm

.macro fmla_col_4x2x4 sum_q0, sum_q1, col0
     fmla    \sum_q0, VSRC_4S_B0, \col0[0]
     fmla    \sum_q0, VSRC_4S_B1, \col0[1]
     fmla    \sum_q1, VSRC_4S_B0, \col0[2]
     fmla    \sum_q1, VSRC_4S_B1, \col0[3]
.endm

.macro fmla_col_4x1x4 sum_q, col0, idx
     fmla    \sum_q, VSRC_4S_B0, \col0[\idx]
.endm

/* void fmul_4xkx4_asm(const float *A, const float *B, float *C, uint32_t K, uint32_t CStride, uint32_t *pSparseFlag) */

	.text
	.align 5
#ifdef __APPLE__
	.global _fmul_4xkx4_asm
_fmul_4xkx4_asm:
#else
	.global fmul_4xkx4_asm
fmul_4xkx4_asm:
#endif

	prfm PLDL1KEEP, [A, #64]
	sxtw CSTRIDEX, CSTRIDE
	eor VSRC_16B_C0, VSRC_16B_C0, VSRC_16B_C0
	lsr KDIV4, K, #2
	prfm PLDL1KEEP, [B, #64]
	eor VSRC_16B_C1, VSRC_16B_C1, VSRC_16B_C1
	and KHAS2, K, #2
	eor VSRC_16B_C2, VSRC_16B_C2, VSRC_16B_C2
	and KHAS1, K, #1
	eor VSRC_16B_C3, VSRC_16B_C3, VSRC_16B_C3

__LOOP_4_BEG:
	cmp KDIV4, #0
	beq __2_BEG

	ld1  {VSRC_4S_A0, VSRC_4S_A1}, [A], #32
    ld1  {VSRC_4S_B0, VSRC_4S_B1, VSRC_4S_B2, VSRC_4S_B3}, [B], #64

    prfm PLDL1KEEP, [A, #64]
	fmla_col_4x4x4 VSRC_4S_C0, VSRC_S_A0
	fmla_col_4x4x4 VSRC_4S_C1, VSRC_S_A1
	ld1  {VSRC_4S_A2, VSRC_4S_A3}, [A], #32

	prfm PLDL1KEEP, [B, #64]
	subs KDIV4, KDIV4, #1
	fmla_col_4x4x4 VSRC_4S_C2, VSRC_S_A2
	fmla_col_4x4x4 VSRC_4S_C3, VSRC_S_A3

	b __LOOP_4_BEG

__2_BEG:
	cmp KHAS2, #0
	beq __1_BEG

	ld1  {VSRC_4S_A0, VSRC_4S_A1}, [A], #32
    ld1  {VSRC_4S_B0, VSRC_4S_B1}, [B], #32
    
    prfm PLDL1KEEP, [A, #64]
	fmla_col_4x2x4 VSRC_4S_C0, VSRC_4S_C1, VSRC_S_A0
	prfm PLDL1KEEP, [B, #64]
	fmla_col_4x2x4 VSRC_4S_C2, VSRC_4S_C3, VSRC_S_A1

__1_BEG:
	cmp KHAS1, #0
	beq __END

	ld1  {VSRC_4S_A0}, [A]
    ld1  {VSRC_4S_B0}, [B]

    fmla_col_4x1x4 VSRC_4S_C0, VSRC_S_A0, 0
    fmla_col_4x1x4 VSRC_4S_C1, VSRC_S_A0, 1
    fmla_col_4x1x4 VSRC_4S_C2, VSRC_S_A0, 2
    fmla_col_4x1x4 VSRC_4S_C3, VSRC_S_A0, 3

__END:
	st1 {VSRC_4S_C0}, [C], CSTRIDEX
	st1 {VSRC_4S_C1}, [C], CSTRIDEX
	st1 {VSRC_4S_C2}, [C], CSTRIDEX
	st1 {VSRC_4S_C3}, [C]

	ret
#endif