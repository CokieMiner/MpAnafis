//! `x86_64` AVX2 out-of-place left-shift kernel.
//!
//! 256-bit loop (`vpsllq`/`vpsrlq`/`vpor`) processing four limbs per
//! iteration with overlapping 256-bit window loads and packed stores,
//! selected at runtime only when the host (including any VM the process runs
//! in) reports AVX2 via CPUID. An SSE2 pair tail covers the last two limbs
//! and a scalar step the last one.

#![expect(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics require typed `__m128i`/`__m256i` pointers while guaranteeing unaligned access; the cast from `Limb` pointers is the intrinsic API itself and asserts no alignment."
)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_sll_epi64, _mm256_srl_epi64,
    _mm256_storeu_si256, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi64x, _mm_sll_epi64,
    _mm_srl_epi64, _mm_storeu_si128,
};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] << shift` (merged across limb
/// boundaries, `0 < shift < LIMB_BITS`). Returns `src[len-1] >> (64-shift)`.
///
/// # Safety
///
/// The CPU must support AVX2. For nonzero `len`, aligned, disjoint spans must
/// cover `len` limbs within `isize::MAX` bytes. Source limbs are initialized;
/// destination limbs are writable and may be uninitialized. `0 < shift < 64`.
#[target_feature(enable = "avx2")]
pub unsafe fn lshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    // SAFETY: the caller provides disjoint readable/writable spans and AVX2.
    // len > 0 and 0 < shift < 64 prove both subtractions exact. Each quad
    // requires index+3 < len; the pair tail requires index+1 < len. Their
    // increments leave index <= len, and every merge reads index-1 >= 0.
    unsafe {
        if len == 0 {
            return 0;
        }
        // Kernel contract: 0 < shift < LIMB_BITS, so 64 - shift cannot
        // underflow; `add(len)` lands one past the end, `sub(1)` on the top
        // limb.
        let drop = 64_u32.unchecked_sub(shift);
        let carry_out = *src.add(len).sub(1) >> drop;
        *dst = *src << shift;
        let left = _mm_set1_epi64x(i64::from(shift));
        let right = _mm_set1_epi64x(i64::from(drop));

        // Each output quad at `index` is
        // (src[index..index + 4] << shift) | (src[index - 1..index + 3] >> drop).
        // Bound: index + 3 < len keeps the quad and its lower merge in
        // bounds, so the step below cannot overflow.
        let mut index = 1;
        if len >= 5 {
            while index < len.unchecked_sub(3) {
                let upper = _mm256_loadu_si256(src.add(index).cast::<__m256i>());
                let lower = _mm256_loadu_si256(src.add(index).sub(1).cast::<__m256i>());
                let merged = _mm256_or_si256(
                    _mm256_sll_epi64(upper, left),
                    _mm256_srl_epi64(lower, right),
                );
                _mm256_storeu_si256(dst.add(index).cast::<__m256i>(), merged);
                index = index.unchecked_add(4);
            }
        }
        // SSE2 pair tail, then a possible single limb. Bound: index + 1 < len
        // keeps the pair and its lower merge in bounds.
        if index < len.unchecked_sub(1) {
            let upper = _mm_loadu_si128(src.add(index).cast::<__m128i>());
            let lower = _mm_loadu_si128(src.add(index).sub(1).cast::<__m128i>());
            let merged = _mm_or_si128(_mm_sll_epi64(upper, left), _mm_srl_epi64(lower, right));
            _mm_storeu_si128(dst.add(index).cast::<__m128i>(), merged);
            index = index.unchecked_add(2);
        }
        if index < len {
            // The merge reads src[index - 1], in bounds because index >= 1.
            *dst.add(index) = (*src.add(index) << shift) | (*src.add(index).sub(1) >> drop);
        }
        carry_out
    }
}
