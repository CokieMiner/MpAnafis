//! Portable multiply-subtract limb kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Universal fallback for `submul` — multiply `len` limbs from `src` by
/// `scalar`, subtract the result from `dst`, and return the final
/// `(carry, borrow)` pair. Used on platforms without assembly implementations.
///
/// # Safety
///
/// - `dst` must be valid for reads and writes of `len` elements and `src`
///   for reads of `len` elements.
/// - `dst` and `src` must not overlap, even partially: the kernel reads
///   `src` while it writes `dst`.
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
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    let mut carry: DoubleLimb = 0;
    let mut borrow: Limb = 0;
    let s = scalar as DoubleLimb;
    // SAFETY: Caller guarantees `dst` and `src` have at least `len` elements
    unsafe {
        for i in 0..len {
            let val = *src.add(i);
            let p = ((val as DoubleLimb).wrapping_mul(s)).wrapping_add(carry);
            carry = p >> LIMB_BITS;
            let lo = p as Limb;

            let u_val = *dst.add(i);
            let (diff1, b1) = u_val.overflowing_sub(lo);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            *dst.add(i) = diff2;
            borrow = Limb::from(b1 | b2);
        }
        (carry as Limb, borrow)
    }
}
