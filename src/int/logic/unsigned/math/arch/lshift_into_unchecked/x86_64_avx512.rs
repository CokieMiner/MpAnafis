//! `x86_64` AVX-512 out-of-place left-shift kernel.
//!
//! 512-bit loop (`vpsllq`/`vpsrlq`/`vpor`/`valignq`) processing eight limbs
//! per iteration, selected at runtime only when the host (including any VM
//! the process runs in) reports `avx512f` via CPUID. One `valignq` supplies
//! each block's incoming cross-lane bits from the previous block's discarded
//! high register, so the loop loads each source block once instead of
//! reloading an overlapping window — GMP's `fastsse/lshift-movdqu2.asm`
//! strategy at four times the width. A scalar tail covers the last at most
//! seven limbs.

#![allow(
    clippy::cast_ptr_alignment,
    reason = "loadu/storeu intrinsics require typed `__m512i` pointers while guaranteeing unaligned access; the cast from `Limb` pointers is the intrinsic API itself and asserts no alignment."
)]

use core::arch::x86_64::{
    __m512i, _mm512_alignr_epi64, _mm512_loadu_si512, _mm512_or_si512, _mm512_setzero_si512,
    _mm512_sll_epi64, _mm512_srl_epi64, _mm512_storeu_si512, _mm_set1_epi64x,
};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] << shift` (merged across limb
/// boundaries, `0 < shift < LIMB_BITS`). Returns `src[len-1] >> (64-shift)`.
///
/// # Safety
///
/// The runtime selector installs this kernel only after the host reports
/// AVX-512; the caller must otherwise meet the same span, non-aliasing, and
/// shift preconditions as the portable kernel.
#[target_feature(enable = "avx512f")]
pub unsafe fn lshift_into_unchecked(
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
        // underflow; `add(len).sub(1)` lands on the top limb.
        let drop = 64_u32.wrapping_sub(shift);
        let carry_out = *src.add(len).sub(1) >> drop;
        let left = _mm_set1_epi64x(i64::from(shift));
        let right = _mm_set1_epi64x(i64::from(drop));

        // Each output block at `index` is
        // (src[index..index + 8] << shift) | (src[index - 1..index + 7] >> drop):
        // lanes 1..7 take their incoming bits from this block's own
        // `dropped` register shifted up one lane, and lane 0 takes lane 7 of
        // the previous block's register — exactly `valignq(dropped, previous, 7)`.
        // Bound: len - index >= 8 keeps the load in bounds.
        let mut index = 0;
        let mut previous_dropped = _mm512_setzero_si512();
        while len.wrapping_sub(index) >= 8 {
            let current = _mm512_loadu_si512(src.add(index).cast::<__m512i>());
            let dropped = _mm512_srl_epi64(current, right);
            let incoming = _mm512_alignr_epi64(dropped, previous_dropped, 7);
            let merged = _mm512_or_si512(_mm512_sll_epi64(current, left), incoming);
            _mm512_storeu_si512(dst.add(index).cast::<__m512i>(), merged);
            previous_dropped = dropped;
            index = index.wrapping_add(8);
        }
        if index == 0 {
            // No full block ran, so limb zero has no incoming bits and the
            // scalar tail below may not read src[-1].
            *dst = *src << shift;
            index = 1;
        }
        // Scalar tail of at most seven limbs. Bound: index >= 1 keeps the
        // `index - 1` merge read in bounds, and index < len throughout.
        while index < len {
            *dst.add(index) = (*src.add(index) << shift) | (*src.add(index).sub(1) >> drop);
            index = index.wrapping_add(1);
        }
        carry_out
    }
}
