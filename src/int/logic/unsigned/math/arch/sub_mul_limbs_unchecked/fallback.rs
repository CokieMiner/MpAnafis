//! Portable multiply-subtract limb kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Evaluates `dst_new - (carry + borrow) * B^len = dst_old - src * scalar`.
///
/// Multiplies `src` by `scalar`, subtracts the product from `dst` with borrow
/// propagation, and returns `(carry, borrow)` where `carry` is the high product
/// limb and `borrow in {0, 1}` is the final subtraction borrow.
///
/// # Safety
///
/// - `dst` must cover `len` aligned, initialized limbs under exclusive access.
/// - `src` must cover `len` aligned, initialized limbs under shared access.
/// - `dst[0..len]` and `src[0..len]` must not overlap.
/// - Each span's byte length must fit in `isize`; zero length permits null pointers.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "inlining exposes the fixed limb width; widening is exact on 16-, 32-, and 64-bit targets, and narrowing selects the low product digit or its bounded high digit"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "narrowing selects the low limb modulo the limb base or the high digit, which is smaller than that base"
    )
)]
#[inline(always)]
pub unsafe fn sub_mul_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    scalar: Limb,
) -> (Limb, Limb) {
    if len == 0 {
        return (0, 0);
    }
    let s = scalar as DoubleLimb;
    // SAFETY: len > 0 supplies the first aligned, initialized limb in each
    // disjoint span; all later indices are below len and the byte bound makes
    // their offsets representable. For B = 2^LIMB_BITS, every product plus
    // carry is at most (B - 1)^2 + (B - 1) < B^2 on all supported limb widths.
    unsafe {
        let first = (*src as DoubleLimb).unchecked_mul(s);
        let (first_difference, first_borrow) = (*dst).overflowing_sub(first as Limb);
        *dst = first_difference;
        let mut carry = first >> LIMB_BITS;
        let mut borrow = first_borrow;
        for i in 1..len {
            let val = *src.add(i);
            let p = (val as DoubleLimb).unchecked_mul(s).unchecked_add(carry);
            carry = p >> LIMB_BITS;
            let lo = p as Limb;

            let u_val = *dst.add(i);
            let (difference, next_borrow) = u_val.borrowing_sub(lo, borrow);
            *dst.add(i) = difference;
            borrow = next_borrow;
        }
        (carry as Limb, Limb::from(borrow))
    }
}
