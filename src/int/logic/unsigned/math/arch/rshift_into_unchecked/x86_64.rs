//! `x86_64` out-of-place right-shift kernel.
//!
//! Uses 128-bit SSE2 operations (`psllq`/`psrlq`/`por`, baseline on
//! `x86_64`),
//! processing two limbs per iteration with one aligned output store — the
//! same strategy as GMP's `fastsse/rshift-movdqu2.asm`. The scalar prologue
//! and tail cover the first and last limbs; pairs never read past the end of
//! `src`.

#![allow(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics require typed `__m128i` pointers while guaranteeing unaligned access; the cast from `Limb` pointers is the intrinsic API itself and asserts no alignment."
)]

use core::arch::x86_64::{
    __m128i, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi64x, _mm_sll_epi64, _mm_srl_epi64,
    _mm_storeu_si128,
};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] >> shift` (merged across limb
/// boundaries, `0 < shift < LIMB_BITS`). Returns `src[0] << (64-shift)`, the
/// bits shifted out of the bottom limb.
///
/// # Safety
///
/// - `dst` must be valid for writes of `len` elements and `src` for reads of
///   `len` elements.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`: the kernel computes
///   `LIMB_BITS - shift`, so an out-of-range amount is undefined behavior.
/// - `dst` and `src` must not overlap, even partially: the kernel reads
///   `src` while it writes `dst`.
#[allow(
    clippy::inline_always,
    reason = "Critical for peak assembly performance"
)]
#[inline(always)]
pub unsafe fn rshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    // SAFETY: Caller guarantees `dst` writable and `src` readable for `len`
    // elements, shift in 1..63, and no aliasing between the spans.
    unsafe {
        if len == 0 {
            return 0;
        }
        // Kernel contract: 0 < shift < LIMB_BITS, so 64 - shift cannot
        // underflow.
        let drop = 64_u32.wrapping_sub(shift);
        let carry_out = *src << drop;
        let mut index = 1_usize;
        if len > 1 {
            *dst = (*src >> shift) | (*src.add(1) << drop);

            let left = _mm_set1_epi64x(i64::from(drop));
            let right = _mm_set1_epi64x(i64::from(shift));
            // Each output pair (index, index + 1) is
            // (src[index..index + 2] >> shift) | (src[index + 1..index + 3] << drop),
            // both source pairs overlapping by one limb and fully in bounds.
            // Bound: index + 2 < len, so the step below cannot overflow.
            while index < len.wrapping_sub(2) {
                let lower = _mm_loadu_si128(src.add(index).cast::<__m128i>());
                let upper = _mm_loadu_si128(src.add(index).add(1).cast::<__m128i>());
                let merged = _mm_or_si128(_mm_srl_epi64(lower, right), _mm_sll_epi64(upper, left));
                _mm_storeu_si128(dst.add(index).cast::<__m128i>(), merged);
                index = index.wrapping_add(2);
            }
            // Bound: index + 1 < len keeps the src[index + 1] merge read in
            // bounds.
            if index < len.wrapping_sub(1) {
                *dst.add(index) = (*src.add(index) >> shift) | (*src.add(index).add(1) << drop);
                index = index.wrapping_add(1);
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
