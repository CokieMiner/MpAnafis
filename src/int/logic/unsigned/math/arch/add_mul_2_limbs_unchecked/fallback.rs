//! Portable fused dual-row multiply-add kernel.

use super::{DoubleLimb, LIMB_BITS, Limb};

/// Accumulate two overlapping scalar-product rows in one source traversal.
///
/// # Safety
///
/// `src` must be valid for `len` readable limbs and `dst` for `len + 1`
/// readable and writable limbs. The source must not overlap the destination.
/// `len == 0` performs no pointer access.
#[allow(
    clippy::inline_always,
    reason = "The fused dual-row loop is a multiplication basecase hot path on generic targets"
)]
#[inline(always)]
pub unsafe fn add_mul_2_limbs_unchecked(
    dst: *mut Limb,
    src: *const Limb,
    len: usize,
    low_scalar: Limb,
    high_scalar: Limb,
) -> (Limb, Limb) {
    let mut low_carry: Limb = 0;
    let mut high_carry: Limb = 0;
    #[allow(clippy::as_conversions, reason = "Limb fits in DoubleLimb")]
    let low_scalar_wide = low_scalar as DoubleLimb;
    #[allow(clippy::as_conversions, reason = "Limb fits in DoubleLimb")]
    let high_scalar_wide = high_scalar as DoubleLimb;

    for i in 0..len {
        // SAFETY: caller guarantees src is valid for `len` elements and dst for `len+1`.
        let source_limb = unsafe { *src.add(i) };
        #[allow(clippy::as_conversions, reason = "Limb fits in DoubleLimb")]
        let source_wide = source_limb as DoubleLimb;

        // SAFETY: The bounds check is guaranteed by the caller. Math overflow is caught and wrapped.
        unsafe {
            #[allow(clippy::as_conversions, reason = "Limb fits in DoubleLimb")]
            let low_carry_wide = low_carry as DoubleLimb;
            #[allow(clippy::as_conversions, reason = "Limb fits in DoubleLimb")]
            let low_destination_wide = (*dst.add(i)) as DoubleLimb;
            let low_sum = source_wide
                .wrapping_mul(low_scalar_wide)
                .wrapping_add(low_carry_wide)
                .wrapping_add(low_destination_wide);

            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "The low LIMB_BITS bits are exactly the result limb"
            )]
            let low_result = low_sum as Limb;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "The sum is below B^2, so its high LIMB_BITS bits fit in Limb"
            )]
            let next_low_carry = (low_sum >> LIMB_BITS) as Limb;
            *dst.add(i) = low_result;
            low_carry = next_low_carry;

            #[allow(clippy::as_conversions, reason = "Limb fits in DoubleLimb")]
            let high_carry_wide = high_carry as DoubleLimb;
            #[allow(clippy::as_conversions, reason = "Limb fits in DoubleLimb")]
            let high_destination_wide = (*dst.add(i.wrapping_add(1))) as DoubleLimb;
            let high_sum = source_wide
                .wrapping_mul(high_scalar_wide)
                .wrapping_add(high_carry_wide)
                .wrapping_add(high_destination_wide);

            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "The low LIMB_BITS bits are exactly the result limb"
            )]
            let high_result = high_sum as Limb;
            #[allow(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "The sum is below B^2, so its high LIMB_BITS bits fit in Limb"
            )]
            let next_high_carry = (high_sum >> LIMB_BITS) as Limb;
            *dst.add(i.wrapping_add(1)) = high_result;
            high_carry = next_high_carry;
        }
    }
    (low_carry, high_carry)
}
