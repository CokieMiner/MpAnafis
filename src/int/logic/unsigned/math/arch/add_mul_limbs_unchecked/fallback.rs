//! Portable multiply-add limb kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Accumulates `src * scalar` into `dst` and returns the high carry limb.
///
/// Empty spans return zero without accessing pointers.
///
/// # Safety
///
/// Nonempty spans must cover `len` aligned, initialized limbs in disjoint live
/// allocations of at most `isize::MAX` bytes. The destination must be writable.
#[expect(
    clippy::inline_always,
    clippy::as_conversions,
    reason = "Inline the widened row recurrence; casts split its low limb and bounded high carry"
)]
#[cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "The low half is reduced modulo the limb base and the shifted high half fits a limb"
    )
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
    // SAFETY: i < len bounds both aligned reads and the writable destination
    // access in the disjoint initialized spans. For B = 2^LIMB_BITS, each
    // widened sum is at most (B-1)^2 + 2*(B-1) = B^2-1; no arithmetic overflows.
    unsafe {
        for i in 0..len {
            let d = *dst.add(i);
            let sr = *src.add(i);
            carry = (d as DoubleLimb)
                .unchecked_add((sr as DoubleLimb).unchecked_mul(s))
                .unchecked_add(carry);
            // DoubleLimb holds at least two limbs on every supported pointer
            // width. The low half reduces modulo B; the shifted high half fits.
            *dst.add(i) = carry as Limb;
            carry >>= LIMB_BITS;
        }
    }
    carry as Limb
}
