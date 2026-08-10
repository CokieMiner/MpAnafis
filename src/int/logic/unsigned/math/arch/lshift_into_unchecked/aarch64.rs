//! `AArch64` out-of-place left-shift kernel.
//!
//! NEON 128-bit operations (`shl`/`ushr`/`orr` via `vshlq_u64` with a
//! positive/negative count vector) process two limbs per iteration. NEON is
//! mandatory on every `AArch64` implementation, so no runtime dispatch is
//! needed. The scalar prologue and tail cover the first and last limbs.

use core::arch::aarch64::{vdupq_n_s64, vld1q_u64, vorrq_u64, vshlq_u64, vst1q_u64};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] << shift` (merged across limb
/// boundaries, `0 < shift < 64`). Returns `src[len-1] >> (64-shift)`.
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

        let left = vdupq_n_s64(i64::from(shift));
        // Negative counts shift right; this is the variable-count form of the
        // `ushr` merge because `extr` requires an immediate. `drop` is in
        // 1..=63 (shift in 1..=63), so `wrapping_neg` equals exact negation.
        let right = vdupq_n_s64(i64::from(drop).wrapping_neg());
        // Each output pair (index, index + 1) is
        // (src[index..index + 2] << shift) | (src[index - 1..index + 1] >> drop),
        // both source pairs overlapping by one limb. Bound: index + 1 < len
        // keeps both pairs in bounds, so the step below cannot overflow.
        let mut index = 1_usize;
        while index < len.wrapping_sub(1) {
            let upper = vld1q_u64(src.add(index).cast::<u64>());
            let lower = vld1q_u64(src.add(index).sub(1).cast::<u64>());
            let merged = vorrq_u64(vshlq_u64(upper, left), vshlq_u64(lower, right));
            vst1q_u64(dst.add(index).cast::<u64>(), merged);
            index = index.wrapping_add(2);
        }
        if index < len {
            // The merge reads src[index - 1], in bounds because index >= 1.
            *dst.add(index) = (*src.add(index) << shift) | (*src.add(index).sub(1) >> drop);
        }
        carry_out
    }
}
