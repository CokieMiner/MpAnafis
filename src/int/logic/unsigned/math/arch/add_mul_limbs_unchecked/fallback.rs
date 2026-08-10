//! Portable multiply-add limb kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Multiply `len` limbs from `src` by `scalar`, add the result into `dst`,
/// and return the final carry.
///
/// This computes:
///
/// ```text
///   (carry, dst[0..len]) = dst[0..len] + (src[0..len] × scalar)
/// ```
///
/// This is the universal fallback implementation used on architectures where no
/// inline assembly optimizations are available.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements.
/// - `src` must be valid for reads of `len` elements.
#[allow(
    clippy::inline_always,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Critical for peak performance on generic platforms; \
              Limb→DoubleLimb is always a widening cast on all targets \
              (u16→u32, u32→u64, u64→u128); DoubleLimb→Limb truncation \
              is correct — the low LIMB_BITS bits are the result limb and \
              the high bits form the carry; arithmetic is bounded by \
              caller-provided lengths"
)]
#[inline(always)]
pub unsafe fn add_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> Limb {
    let mut carry: DoubleLimb = 0;
    let s = scalar as DoubleLimb;
    // SAFETY: Caller guarantees that `dst` and `src` have at least `len` limbs
    unsafe {
        for i in 0..len {
            let d = *dst.add(i);
            let sr = *src.add(i);
            carry = (d as DoubleLimb)
                .wrapping_add((sr as DoubleLimb).wrapping_mul(s))
                .wrapping_add(carry);
            *dst.add(i) = carry as Limb;
            carry >>= LIMB_BITS;
        }
    }
    carry as Limb
}
