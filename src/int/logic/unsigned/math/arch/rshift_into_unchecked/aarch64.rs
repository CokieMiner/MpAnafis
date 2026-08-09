//! `AArch64` out-of-place right-shift kernel.
//!
//! NEON 128-bit operations (`shl`/`ushr`/`orr` via `vshlq_u64` with a
//! positive/negative count vector) process two limbs per iteration. NEON is
//! mandatory on every `AArch64` implementation, so no runtime dispatch is
//! needed. The scalar prologue and tail cover the first and last limbs;
//! pairs never read past the end of `src`.

use core::arch::aarch64::{vdupq_n_s64, vld1q_u64, vorrq_u64, vshlq_u64, vst1q_u64};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] >> shift` (merged across limb
/// boundaries, `0 < shift < 64`). Returns `src[0] << (64-shift)`, the bits
/// shifted out of the bottom limb.
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

            let left = vdupq_n_s64(i64::from(drop));
            // Negative counts shift right; this is the variable-count form of
            // the `ushr` merge because `extr` requires an immediate. `shift`
            // is in 1..=63, so `wrapping_neg` equals exact negation.
            let right = vdupq_n_s64(i64::from(shift).wrapping_neg());
            // Each output pair (index, index + 1) is
            // (src[index..index + 2] >> shift) | (src[index + 1..index + 3] << drop),
            // both source pairs overlapping by one limb and fully in bounds.
            // Bound: index + 2 < len, so the step below cannot overflow.
            while index < len.wrapping_sub(2) {
                let lower = vld1q_u64(src.add(index).cast::<u64>());
                let upper = vld1q_u64(src.add(index).add(1).cast::<u64>());
                let merged = vorrq_u64(vshlq_u64(lower, right), vshlq_u64(upper, left));
                vst1q_u64(dst.add(index).cast::<u64>(), merged);
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
