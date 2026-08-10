//! `x86_64` AVX2 in-place right-shift kernel.
//!
//! 256-bit loop (`vpsllq`/`vpsrlq`/`vpor`) processing four limbs per
//! iteration, selected at runtime only when the host (including any VM the
//! process runs in) reports AVX2 via CPUID. The loop ascends; each quad
//! writes positions index..index + 4 while the next iteration reads from
//! index + 4..index + 8, so the writes never reach an unread source limb.
//! A scalar tail covers the high limbs.

#![allow(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics require typed `__m128i`/`__m256i` pointers while guaranteeing unaligned access; the cast from `Limb` pointers is the intrinsic API itself and asserts no alignment."
)]

use core::arch::x86_64::{
    __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_sll_epi64, _mm256_srl_epi64,
    _mm256_storeu_si256, _mm_set1_epi64x,
};

use super::Limb;

/// Right-shift `len` limbs in-place by `shift` bits (0 < shift < `LIMB_BITS`).
/// Returns the bits shifted out of the bottom limb.
///
/// # Safety
///
/// The runtime selector installs this kernel only after the host reports
/// AVX2; callers must otherwise meet the same span and shift preconditions as
/// the portable kernel.
#[target_feature(enable = "avx2")]
pub unsafe fn rshift_unchecked(limbs: *mut Limb, len: usize, shift: u32) -> Limb {
    // SAFETY: Caller guarantees `limbs` has `len` elements and shift in
    // 1..63; the runtime selector proved AVX2 before installing this kernel.
    unsafe {
        if len == 0 {
            return 0;
        }
        // Kernel contract: 0 < shift < LIMB_BITS, so 64 - shift cannot
        // underflow.
        let drop = 64_u32.wrapping_sub(shift);
        let carry_out = *limbs << drop;

        // Quad at `index` writes positions index..index + 4 from sources
        // index..index + 4 and index + 1..index + 5. Ascending by 4 leaves
        // position index + 4, the first limb the next quad reads, untouched.
        let mut index = 0;
        if len >= 5 {
            let left = _mm_set1_epi64x(i64::from(drop));
            let right = _mm_set1_epi64x(i64::from(shift));
            // Bound: index + 4 < len keeps the upper read, index + 1..index + 5
            // within the span, so the step below cannot overflow.
            while index < len.wrapping_sub(4) {
                let lower = _mm256_loadu_si256(limbs.add(index).cast::<__m256i>());
                let upper = _mm256_loadu_si256(limbs.add(index).add(1).cast::<__m256i>());
                let merged = _mm256_or_si256(
                    _mm256_srl_epi64(lower, right),
                    _mm256_sll_epi64(upper, left),
                );
                _mm256_storeu_si256(limbs.add(index).cast::<__m256i>(), merged);
                index = index.wrapping_add(4);
            }
        }

        // Scalar tail for positions index..len (at most 4 limbs), ascending:
        // out[i] = (src[i] >> shift) | (src[i + 1] << drop), then the bare
        // top limb. Bound: index + 1 < len keeps index + 1 within the span.
        while index < len.wrapping_sub(1) {
            *limbs.add(index) = (*limbs.add(index) >> shift) | (*limbs.add(index).add(1) << drop);
            index = index.wrapping_add(1);
        }
        if index < len {
            *limbs.add(index) = *limbs.add(index) >> shift;
        }
        carry_out
    }
}
