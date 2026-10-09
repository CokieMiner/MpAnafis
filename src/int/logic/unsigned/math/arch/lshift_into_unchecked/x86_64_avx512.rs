//! `x86_64` AVX-512 out-of-place left-shift kernel.
//!
//! 512-bit loop (`vpsllq`/`vpsrlq`/`vpor`/`valignq`) processing eight limbs
//! per iteration, selected at runtime only when the host (including any VM
//! the process runs in) reports `avx512f` via CPUID. One `valignq` supplies
//! each block's incoming cross-lane bits from the previous block's discarded
//! high register, so the loop loads each source block once instead of
//! reloading an overlapping window, using cross-iteration register forwarding
//! to eliminate redundant cache loads. A scalar tail covers the last at most
//! seven limbs.

#![expect(
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
/// The CPU must support AVX-512F. For nonzero `len`, aligned, disjoint spans
/// must cover `len` limbs within `isize::MAX` bytes. Source limbs are initialized;
/// destination limbs are writable and may be uninitialized. `0 < shift < 64`.
#[target_feature(enable = "avx512f")]
pub unsafe fn lshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    // SAFETY: the caller provides disjoint readable/writable spans and AVX-512F.
    // len > 0 and 0 < shift < 64 prove len-1 and 64-shift exact. index starts
    // at zero and increases by eight only when len-index >= 8, so index <= len.
    // All vector accesses stay within the spans; the tail reads index-1 only
    // after index >= 1. Every destination limb is initialized once.
    unsafe {
        if len == 0 {
            return 0;
        }
        // Kernel contract: 0 < shift < LIMB_BITS, so 64 - shift cannot
        // underflow; `add(len).sub(1)` lands on the top limb.
        let drop = 64_u32.unchecked_sub(shift);
        let carry_out = *src.add(len).sub(1) >> drop;
        let left = _mm_set1_epi64x(i64::from(shift));
        let right = _mm_set1_epi64x(i64::from(drop));

        // Each output block at `index` is
        // (src[index..index + 8] << shift) | (src[index - 1..index + 7] >> drop):
        // lanes 1..7 take their incoming bits from this block's own
        // `dropped` register shifted up one lane, and lane 0 takes lane 7 of
        // the previous block's register: exactly `valignq(dropped, previous, 7)`.
        // Bound: len - index >= 8 keeps the load in bounds.
        let mut index = 0;
        let mut previous_dropped = _mm512_setzero_si512();
        while len.unchecked_sub(index) >= 8 {
            let current = _mm512_loadu_si512(src.add(index).cast::<__m512i>());
            let dropped = _mm512_srl_epi64(current, right);
            let incoming = _mm512_alignr_epi64(dropped, previous_dropped, 7);
            let merged = _mm512_or_si512(_mm512_sll_epi64(current, left), incoming);
            _mm512_storeu_si512(dst.add(index).cast::<__m512i>(), merged);
            previous_dropped = dropped;
            index = index.unchecked_add(8);
        }
        if index == 0 {
            // No full block ran, so limb zero has no incoming bits and the
            // scalar tail below may not read src[-1].
            *dst = *src << shift;
            index = 1;
        }
        // Scalar tail of at most seven limbs. Bound: index >= 1 keeps the
        // `index - 1` merge read in bounds, and index < len throughout.
        for tail_index in index..len {
            *dst.add(tail_index) = (*src.add(tail_index) << shift) | (*src.add(tail_index).sub(1) >> drop);
        }
        carry_out
    }
}
