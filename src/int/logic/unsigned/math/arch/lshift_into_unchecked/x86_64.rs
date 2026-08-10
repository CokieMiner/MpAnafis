//! `x86_64` out-of-place left-shift kernel.
//!
//! Uses 128-bit SSE2 operations (`psllq`/`psrlq`/`por`, baseline on
//! `x86_64`),
//! processing two limbs per iteration with one aligned output store — the
//! same strategy as GMP's `fastsse/lshift-movdqu2.asm`, which is what keeps
//! large shifts at memory-bandwidth speed. The scalar prologue and tail cover
//! the first and last limbs.

#![allow(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics require typed `__m128i` pointers while guaranteeing unaligned access; the cast from `Limb` pointers is the intrinsic API itself and asserts no alignment."
)]

use core::arch::x86_64::{
    __m128i, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi64x, _mm_sll_epi64, _mm_srl_epi64,
    _mm_storeu_si128,
};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] << shift` (merged across limb
/// boundaries, `0 < shift < LIMB_BITS`). Returns `src[len-1] >> (64-shift)`.
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
pub unsafe fn lshift_into_unchecked(
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
        // underflow; `add(len)` lands one past the end, `sub(1)` on the top
        // limb.
        let drop = 64_u32.wrapping_sub(shift);
        let carry_out = *src.add(len).sub(1) >> drop;
        *dst = *src << shift;

        let mut index = 1_usize;
        let left = _mm_set1_epi64x(i64::from(shift));
        let right = _mm_set1_epi64x(i64::from(drop));
        // Each output pair (index, index + 1) is
        // (src[index..index + 2] << shift) | (src[index - 1..index + 1] >> drop),
        // both source pairs overlapping by one limb. Bound: index + 1 < len
        // keeps both pairs in bounds, so the step below cannot overflow.
        while index < len.wrapping_sub(1) {
            let upper = _mm_loadu_si128(src.add(index).cast::<__m128i>());
            let lower = _mm_loadu_si128(src.add(index).sub(1).cast::<__m128i>());
            let merged = _mm_or_si128(_mm_sll_epi64(upper, left), _mm_srl_epi64(lower, right));
            _mm_storeu_si128(dst.add(index).cast::<__m128i>(), merged);
            index = index.wrapping_add(2);
        }
        if index < len {
            // The merge reads src[index - 1], in bounds because index >= 1
            // (the pair loop advances by 2 and starts at 1).
            *dst.add(index) = (*src.add(index) << shift) | (*src.add(index).sub(1) >> drop);
        }
        carry_out
    }
}
