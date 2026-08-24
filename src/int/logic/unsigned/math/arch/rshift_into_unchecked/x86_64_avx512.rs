//! `x86_64` AVX-512 out-of-place right-shift kernel.
//!
//! 512-bit loop (`vpsrlq`/`vpsllq`/`vpor`/`valignq`) processing eight limbs
//! per iteration from the top of the span down, selected at runtime only when
//! the host reports `avx512f` via CPUID. One `valignq` supplies each block's
//! incoming cross-lane bits from the block above, so the loop loads each
//! source block once instead of reloading an overlapping window — GMP's
//! `fastsse/rshift-movdqu2.asm` strategy at four times the width. A scalar
//! tail covers the lowest at most seven limbs.

#![allow(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics require typed `__m512i` pointers while guaranteeing unaligned access; the cast from `Limb` pointers is the intrinsic API itself and asserts no alignment."
)]

use core::arch::x86_64::{
    __m512i, _mm512_alignr_epi64, _mm512_loadu_si512, _mm512_or_si512, _mm512_setzero_si512,
    _mm512_sll_epi64, _mm512_srl_epi64, _mm512_storeu_si512, _mm_set1_epi64x,
};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] >> shift` (merged across limb
/// boundaries, `0 < shift < LIMB_BITS`). Returns `src[0] << (64-shift)`, the
/// bits shifted out of the bottom limb.
///
/// # Safety
///
/// The runtime selector installs this kernel only after the host reports
/// AVX-512; the caller must otherwise meet the same span, non-aliasing, and
/// shift preconditions as the portable kernel.
#[target_feature(enable = "avx512f")]
pub unsafe fn rshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    // SAFETY: Caller guarantees `dst` writable and `src` readable for `len`
    // elements, shift in 1..63, and no aliasing between the spans; the
    // runtime selector proved AVX-512 before installing this kernel.
    unsafe {
        if len == 0 {
            return 0;
        }
        // Kernel contract: 0 < shift < LIMB_BITS, so 64 - shift cannot
        // underflow.
        let drop = 64_u32.wrapping_sub(shift);
        let carry_out = *src << drop;
        let left = _mm_set1_epi64x(i64::from(drop));
        let right = _mm_set1_epi64x(i64::from(shift));

        // Each output block at `index` is
        // (src[index..index + 8] >> shift) | (src[index + 1..index + 9] << drop):
        // lanes 0..6 take their incoming bits from this block's own `lifted`
        // register shifted down one lane, and lane 7 takes lane 0 of the
        // block above — exactly `valignq(above_lifted, lifted, 1)`. Blocks
        // run from the top down so the block above is always already
        // computed. Bound: index >= 8 keeps the load's eight limbs in
        // bounds.
        let mut index = len;
        let mut above_lifted = _mm512_setzero_si512();
        while index >= 8 {
            index = index.wrapping_sub(8);
            let current = _mm512_loadu_si512(src.add(index).cast::<__m512i>());
            let lifted = _mm512_sll_epi64(current, left);
            let incoming = _mm512_alignr_epi64(above_lifted, lifted, 1);
            let merged = _mm512_or_si512(_mm512_srl_epi64(current, right), incoming);
            _mm512_storeu_si512(dst.add(index).cast::<__m512i>(), merged);
            above_lifted = lifted;
        }
        if index == len {
            // No full block ran, so the top limb has no incoming bits and
            // the scalar tail below may not read src[len].
            index = index.wrapping_sub(1);
            *dst.add(index) = *src.add(index) >> shift;
        }
        // Scalar tail over the lowest limbs. Bound: index <= len - 1 here,
        // and every limb reads src[index + 1] at most one below its block.
        while index > 0 {
            index = index.wrapping_sub(1);
            *dst.add(index) = (*src.add(index) >> shift) | (*src.add(index).add(1) << drop);
        }
        carry_out
    }
}
