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
#define VSRC_SUM_RET  S20
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

/* void tcopy_4x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */

	.text
	.align 5
#ifdef __APPLE__
	.global _tcopy_4x4_asm
_tcopy_4x4_asm:
#else
	.global tcopy_4x4_asm
tcopy_4x4_asm:
#endif
	ld1 {VSRC_4S_0}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_4S_1}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_4S_2}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_4S_3}, [SRC]
	st1 {VSRC_4S_0, VSRC_4S_1, VSRC_4S_2, VSRC_4S_3}, [DST]

	ret

/* void tcopy_2x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _tcopy_2x4_asm
_tcopy_2x4_asm:
#else
	.global tcopy_2x4_asm
tcopy_2x4_asm:
#endif
	ld1 {VSRC_2S_0}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_2S_1}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_2S_2}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_2S_3}, [SRC]
	st1 {VSRC_2S_0, VSRC_2S_1, VSRC_2S_2, VSRC_2S_3}, [DST]

	ret

/* void tcopy_1x4_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _tcopy_1x4_asm
_tcopy_1x4_asm:
#else
	.global tcopy_1x4_asm
tcopy_1x4_asm:
#endif
	ld1 VSRC_1S_0, [SRC]
	add SRC, SRC, STRIDE
	ld1 VSRC_1S_1, [SRC]
	add SRC, SRC, STRIDE
	ld1 VSRC_1S_2, [SRC]
	add SRC, SRC, STRIDE
	ld1 VSRC_1S_3, [SRC]
	st1 {VSRC_4S_0}, [DST]

	ret

/* uint32_t tcopy_4x4_asm_sparse(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */

	.text
	.align 5
#ifdef __APPLE__
	.global _tcopy_4x4_asm_sparse
_tcopy_4x4_asm_sparse:
#else
	.global tcopy_4x4_asm_sparse
tcopy_4x4_asm_sparse:
#endif
	ld1 {VSRC_4S_0}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_4S_1}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_4S_2}, [SRC]
	and VSRC_16B_SUM, VSRC_16B_0, VSRC_16B_1
	add SRC, SRC, STRIDE
	ld1 {VSRC_4S_3}, [SRC]
	and VSRC_16B_SUM, VSRC_16B_SUM, VSRC_16B_2
	st1 {VSRC_4S_0, VSRC_4S_1, VSRC_4S_2, VSRC_4S_3}, [DST]
	and VSRC_16B_SUM, VSRC_16B_SUM, VSRC_16B_3
	addv VSRC_SUM_RET, VSRC_4S_SUM

	mov w0, VSRC_SUM_S0
	ret

/* uint32_t tcopy_2x4_asm_sparse(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _tcopy_2x4_asm_sparse
_tcopy_2x4_asm_sparse:
#else
	.global tcopy_2x4_asm_sparse
tcopy_2x4_asm_sparse:
#endif
	ld1 {VSRC_2S_0}, [SRC]
	eor VSRC_16B_SUM, VSRC_16B_SUM, VSRC_16B_SUM
	add SRC, SRC, STRIDE
	ld1 {VSRC_2S_1}, [SRC]
	add SRC, SRC, STRIDE
	and VSRC_8B_SUM, VSRC_8B_0, VSRC_8B_1
	ld1 {VSRC_2S_2}, [SRC]
	add SRC, SRC, STRIDE
	and VSRC_8B_SUM, VSRC_8B_SUM, VSRC_8B_2
	ld1 {VSRC_2S_3}, [SRC]
	and VSRC_8B_SUM, VSRC_8B_SUM, VSRC_8B_3
	st1 {VSRC_2S_0, VSRC_2S_1, VSRC_2S_2, VSRC_2S_3}, [DST]
	addv VSRC_SUM_RET, VSRC_4S_SUM

	mov w0, VSRC_SUM_S0
	ret

/* uint32_t tcopy_1x4_asm_sparse(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _tcopy_1x4_asm_sparse
_tcopy_1x4_asm_sparse:
#else
	.global tcopy_1x4_asm_sparse
tcopy_1x4_asm_sparse:
#endif
	ld1 VSRC_1S_0, [SRC]
	add SRC, SRC, STRIDE
	ld1 VSRC_1S_1, [SRC]
	add SRC, SRC, STRIDE
	ld1 VSRC_1S_2, [SRC]
	add SRC, SRC, STRIDE
	ld1 VSRC_1S_3, [SRC]
	addv VSRC_SUM_RET, VSRC_4S_0
	st1 {VSRC_4S_0}, [DST]

	mov w0, VSRC_SUM_S0
	ret






/* void tcopy_4x2_asm(uint32_t *pSrc, uint32_t stride, uint32_t *pDst) */
	.text
	.align 5
#ifdef __APPLE__
	.global _tcopy_4x2_asm
_tcopy_4x2_asm:
#else
	.global tcopy_4x2_asm
tcopy_4x2_asm:
#endif
	ld1 {VSRC_4S_0}, [SRC]
	add SRC, SRC, STRIDE
	ld1 {VSRC_4S_1}, [SRC]
	st1 {VSRC_4S_0, VSRC_4S_1}, [DST]

	ret
#endif