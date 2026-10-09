//! `x86_64` AVX2 out-of-place right-shift kernel.
//!
//! 256-bit loop (`vpsllq`/`vpsrlq`/`vpor`) processing four limbs per
//! iteration with overlapping 256-bit window loads and packed stores,
//! selected at runtime only when the host (including any VM the process runs
//! in) reports AVX2 via CPUID. An SSE2 pair tail and scalar steps cover the
//! last limbs; pairs never read past the end of `src`.

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

/// Writes `dst[0..len] = src[0..len] >> shift` (merged across limb
/// boundaries, `0 < shift < LIMB_BITS`). Returns `src[0] << (64-shift)`, the
/// bits shifted out of the bottom limb.
///
/// # Safety
///
/// The CPU must support AVX2. For nonzero `len`, aligned, disjoint spans must
/// cover `len` limbs within `isize::MAX` bytes. Source limbs are initialized;
/// destination limbs are writable and may be uninitialized. `0 < shift < 64`.
#[target_feature(enable = "avx2")]
pub unsafe fn rshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    // SAFETY: the caller establishes readable/writable disjoint spans and AVX2.
    // len > 0 and 0 < shift < 64 make the complementary count exact. Quads
    // run only for len >= 6 and index+4 < len; pairs require index+2 < len.
    // Their advances stay below len, and every scalar access is similarly bounded.
    unsafe {
        if len == 0 {
            return 0;
        }
        // Kernel contract: 0 < shift < LIMB_BITS, so 64 - shift cannot
        // underflow.
        let drop = 64_u32.unchecked_sub(shift);
        let carry_out = *src << drop;
        let mut index = 1;
        if len > 1 {
            *dst = (*src >> shift) | (*src.add(1) << drop);
            let left = _mm_set1_epi64x(i64::from(drop));
            let right = _mm_set1_epi64x(i64::from(shift));

            // Each output quad at `index` is
            // (src[index..index + 4] >> shift) | (src[index + 1..index + 5] << drop),
            // fully in bounds because the loop leaves at least one source
            // limb beyond the upper read. Bound: index + 4 < len, so the
            // step below cannot overflow.
            if len >= 6 {
                while index < len.unchecked_sub(4) {
                    let lower = _mm256_loadu_si256(src.add(index).cast::<__m256i>());
                    let upper = _mm256_loadu_si256(src.add(index).add(1).cast::<__m256i>());
                    let merged = _mm256_or_si256(
                        _mm256_srl_epi64(lower, right),
                        _mm256_sll_epi64(upper, left),
                    );
                    _mm256_storeu_si256(dst.add(index).cast::<__m256i>(), merged);
                    index = index.unchecked_add(4);
                }
            }
            // SSE2 pair tail, then up to two single limbs. Bound:
            // index + 2 < len keeps the upper pair fully in bounds.
            if index < len.unchecked_sub(2) {
                let lower = _mm_loadu_si128(src.add(index).cast::<__m128i>());
                let upper = _mm_loadu_si128(src.add(index).add(1).cast::<__m128i>());
                let merged = _mm_or_si128(_mm_srl_epi64(lower, right), _mm_sll_epi64(upper, left));
                _mm_storeu_si128(dst.add(index).cast::<__m128i>(), merged);
                index = index.unchecked_add(2);
            }
            // Bound: index + 1 < len keeps the src[index + 1] merge read in
            // bounds.
            if index < len.unchecked_sub(1) {
                *dst.add(index) = (*src.add(index) >> shift) | (*src.add(index).add(1) << drop);
                index = index.unchecked_add(1);
            }
            if index < len {
                *dst.add(index) = *src.add(index) >> shift;
            }
        } else {
            *dst = *src >> shift;
        }
        carry_out
    }
}
