//! `AArch64` out-of-place left-shift kernel.
//!
//! NEON 128-bit operations (`shl`/`ushr`/`orr` via `vshlq_u64` with a
//! positive/negative count vector) process two limbs per iteration. NEON is
//! enabled by the supported `AArch64` targets, so no runtime dispatch is
//! needed. The scalar prologue and tail cover the first and last limbs.

use core::arch::aarch64::{vdupq_n_s64, vld1q_u64, vorrq_u64, vshlq_u64, vst1q_u64};

use super::Limb;

/// Writes `dst[0..len] = src[0..len] << shift` (merged across limb
/// boundaries, `0 < shift < 64`). Returns `src[len-1] >> (64-shift)`.
///
/// # Safety
///
/// - `src` covers `len` aligned initialized limbs; `dst` covers `len` aligned
///   writable limbs, which may be uninitialized. Both byte spans fit in `isize`.
/// - Zero length permits null pointers; the target must enable NEON.
/// - `shift` must satisfy `0 < shift < LIMB_BITS`: the kernel computes
///   `LIMB_BITS - shift`, so an out-of-range amount is undefined behavior.
/// - `dst` and `src` must not overlap, even partially: the kernel reads
///   `src` while it writes `dst`.
#[expect(
    clippy::inline_always,
    reason = "Inlining keeps the NEON loop at its arithmetic caller"
)]
#[inline(always)]
pub unsafe fn lshift_into_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    shift: u32,
) -> Limb {
    // SAFETY: aligned source and write-only destination cover len disjoint
    // limbs. For len > 0, len-1 and 64-shift are exact. Pair iterations require
    // index+1 < len and advance index by two to at most len; the scalar tail
    // accesses index < len and index-1 >= 0. NEON is enabled by the target.
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

        let left = vdupq_n_s64(i64::from(shift));
        // Negative counts shift right; this is the variable-count form of the
        // `ushr` merge because `extr` requires an immediate. `drop` is in
        // 1..=63, so subtracting drop from signed zero cannot overflow.
        let right = vdupq_n_s64(0_i64.unchecked_sub(i64::from(drop)));
        // Each output pair (index, index + 1) is
        // (src[index..index + 2] << shift) | (src[index - 1..index + 1] >> drop),
        // both source pairs overlapping by one limb. Bound: index + 1 < len
        // keeps both pairs in bounds, so the step below cannot overflow.
        let mut index = 1_usize;
        while index < len.unchecked_sub(1) {
            let upper = vld1q_u64(src.add(index).cast::<u64>());
            let lower = vld1q_u64(src.add(index).sub(1).cast::<u64>());
            let merged = vorrq_u64(vshlq_u64(upper, left), vshlq_u64(lower, right));
            vst1q_u64(dst.add(index).cast::<u64>(), merged);
            index = index.unchecked_add(2);
        }
        if index < len {
            // The merge reads src[index - 1], in bounds because index >= 1.
            *dst.add(index) = (*src.add(index) << shift) | (*src.add(index).sub(1) >> drop);
        }
        carry_out
    }
}
