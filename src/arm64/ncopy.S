#ifdef __aarch64__
/* RSV X19~X28 */
/**************in param**************/
#define SRC 		   x0
#define STRIDE         x1
#define DST 		   x2

/* RSV V8~V15 */
#define VSRC_4S_0     V0.4S
#define VSRC_16B_0    V0.16B
#define VSRC_4S_1     V1.4S
#define VSRC_16B_1    V1.16B
#define VSRC_4S_2     V2.4S
#define VSRC_16B_2    V2.16B
#define VSRC_4S_3     V3.4S
#define VSRC_16B_3    V3.16B

#define VSRC_2S_SUM   V4.2S
#define VSRC_4S_SUM   V4.4S
#define VSRC_16B_SUM  V4.16B
#define VSRC_8B_SUM   V4.8B
#define VSRC_SUM_RET  S5
#define VSRC_SUM_S0   V5.S[0]

#define VSRC_2S_0     V0.2S
#define VSRC_8B_0     V0.8B
#define VSRC_2S_1     V1.2S
#define VSRC_8B_1     V1.8B
#define VSRC_2S_2     V2.2S
#define VSRC_8B_2     V2.8B
#define VSRC_2S_3     V3.2S
#define VSRC_8B_3     V3.8B

#define VSRC_1S_0     {V0.S}[0]
#define VSRC_1S_1     {V0.S}[1]
#define VSRC_1S_2     {V0.S}[2]
#define VSRC_1S_3     {V0.S}[3]

#define NO_NEED_TRANSPORT

/* uint32_t ncopy_4x4_asm_sparse(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */

	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_4x4_asm_sparse
_ncopy_4x4_asm_sparse:
#else
	.global ncopy_4x4_asm_sparse
ncopy_4x4_asm_sparse:
#endif
	ld1 {VSRC_4S_0}, [SRC], STRIDE
	ld1 {VSRC_4S_1}, [SRC], STRIDE
	and VSRC_16B_SUM, VSRC_16B_0, VSRC_16B_1
	ld1 {VSRC_4S_2}, [SRC], STRIDE
	and VSRC_16B_SUM, VSRC_16B_SUM, VSRC_16B_2
	ld1 {VSRC_4S_3}, [SRC]
	and VSRC_16B_SUM, VSRC_16B_SUM, VSRC_16B_3
#ifdef NO_NEED_TRANSPORT
	st1 {VSRC_4S_0, VSRC_4S_1, VSRC_4S_2, VSRC_4S_3}, [DST]
#else
	st4 {VSRC_4S_0, VSRC_4S_1, VSRC_4S_2, VSRC_4S_3}, [DST]
#endif
	/* use as uint32_t, as .0f in hex is 0 */
	addv VSRC_SUM_RET, VSRC_4S_SUM

	mov w0, VSRC_SUM_S0
	ret

/* uint32_t ncopy_2x4_asm_sparse(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_2x4_asm_sparse
_ncopy_2x4_asm_sparse:
#else
	.global ncopy_2x4_asm_sparse
ncopy_2x4_asm_sparse:
#endif
	ld1 {VSRC_2S_0}, [SRC], STRIDE
	/* clear high 8B to zero */
	eor VSRC_16B_SUM, VSRC_16B_SUM, VSRC_16B_SUM
	ld1 {VSRC_2S_1}, [SRC], STRIDE
	and VSRC_8B_SUM, VSRC_8B_0, VSRC_8B_1
	ld1 {VSRC_2S_2}, [SRC], STRIDE
	and VSRC_8B_SUM, VSRC_8B_SUM, VSRC_8B_2
	ld1 {VSRC_2S_3}, [SRC]
	and VSRC_8B_SUM, VSRC_8B_SUM, VSRC_8B_3
#ifdef NO_NEED_TRANSPORT
	st1 {VSRC_2S_0, VSRC_2S_1, VSRC_2S_2, VSRC_2S_3}, [DST]
#else
	st4 {VSRC_2S_0, VSRC_2S_1, VSRC_2S_2, VSRC_2S_3}, [DST]
#endif
	addv VSRC_SUM_RET, VSRC_4S_SUM

	mov w0, VSRC_SUM_S0
	ret

/* uint32_t ncopy_1x4_asm_sparse(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_1x4_asm_sparse
_ncopy_1x4_asm_sparse:
#else
	.global ncopy_1x4_asm_sparse
ncopy_1x4_asm_sparse:
#endif
	ld1 VSRC_1S_0, [SRC], STRIDE
	ld1 VSRC_1S_1, [SRC], STRIDE
	ld1 VSRC_1S_2, [SRC], STRIDE
	ld1 VSRC_1S_3, [SRC]
	addv VSRC_SUM_RET, VSRC_4S_0
	st1 {VSRC_4S_0}, [DST]

	mov w0, VSRC_SUM_S0
	ret

/* void ncopy_4x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */

	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_4x4_asm
_ncopy_4x4_asm:
#else
	.global ncopy_4x4_asm
ncopy_4x4_asm:
#endif
	ld1 {VSRC_4S_0}, [SRC], STRIDE
	ld1 {VSRC_4S_1}, [SRC], STRIDE
	ld1 {VSRC_4S_2}, [SRC], STRIDE
	ld1 {VSRC_4S_3}, [SRC]
#ifdef NO_NEED_TRANSPORT
	st1 {VSRC_4S_0, VSRC_4S_1, VSRC_4S_2, VSRC_4S_3}, [DST]
#else
	st4 {VSRC_4S_0, VSRC_4S_1, VSRC_4S_2, VSRC_4S_3}, [DST]
#endif
	ret

/* void ncopy_2x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_2x4_asm
_ncopy_2x4_asm:
#else
	.global ncopy_2x4_asm
ncopy_2x4_asm:
#endif
	ld1 {VSRC_2S_0}, [SRC], STRIDE
	ld1 {VSRC_2S_1}, [SRC], STRIDE
	ld1 {VSRC_2S_2}, [SRC], STRIDE
	ld1 {VSRC_2S_3}, [SRC]
#ifdef NO_NEED_TRANSPORT
	st1 {VSRC_2S_0, VSRC_2S_1, VSRC_2S_2, VSRC_2S_3}, [DST]
#else
	st4 {VSRC_2S_0, VSRC_2S_1, VSRC_2S_2, VSRC_2S_3}, [DST]
#endif
	ret

/* void ncopy_1x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_1x4_asm
_ncopy_1x4_asm:
#else
	.global ncopy_1x4_asm
ncopy_1x4_asm:
#endif
	ld1 VSRC_1S_0, [SRC], STRIDE
	ld1 VSRC_1S_1, [SRC], STRIDE
	ld1 VSRC_1S_2, [SRC], STRIDE
	ld1 VSRC_1S_3, [SRC]
	st1 {VSRC_4S_0}, [DST]

	ret

/* void ncopy_4x2_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_4x2_asm
_ncopy_4x2_asm:
#else
	.global ncopy_4x2_asm
ncopy_4x2_asm:
#endif
	ld1 {VSRC_4S_0}, [SRC], STRIDE
	ld1 {VSRC_4S_1}, [SRC]
#ifdef NO_NEED_TRANSPORT
	st1 {VSRC_4S_0, VSRC_4S_1}, [DST]
#else
	st2 {VSRC_4S_0, VSRC_4S_1}, [DST]
#endif
	ret

/* void ncopy_4x1_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _ncopy_4x1_asm
_ncopy_4x1_asm:
#else
	.global ncopy_4x1_asm
ncopy_4x1_asm:
#endif
	ld1 {VSRC_4S_0}, [SRC]
	st1 {VSRC_4S_0}, [DST]

	ret
#endif