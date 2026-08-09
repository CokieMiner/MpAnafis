//! Reference properties for two-limb-by-one-limb division kernels.

#![allow(
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The properties exercise unsafe hardware contracts; Limb-to-DoubleLimb is widening and quotient/remainder truncation is proven by rem_hi < divisor"
)]

use proptest::prelude::*;

use crate::int::{
    logic::math::arch::ArchKernels,
    types::{DoubleLimb, Limb},
};

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
#[path = "../divrem_1_unchecked/half_limb.rs"]
mod half_limb_test;

proptest! {
    #[test]
    fn prop_selected_divrem_1_matches_double_limb(
        limb in any::<Limb>(),
        rem_seed in any::<Limb>(),
        divisor in any::<Limb>().prop_filter("divisor must be non-zero", |value| *value != 0),
    ) {
        // SAFETY: the strategy filters out zero divisors.
        let (rem_hi, expected) = unsafe { reference_divrem(limb, rem_seed, divisor) };

        // SAFETY: the generated divisor is non-zero and the modulo construction
        // proves rem_hi < divisor.
        let actual = unsafe { ArchKernels::divrem_1_unchecked(limb, rem_hi, divisor) };
        prop_assert_eq!(actual, expected);
    }

    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    #[test]
    fn prop_normalized_half_limb_core_matches_double_limb(
        limb in any::<Limb>(),
        rem_seed in any::<Limb>(),
        divisor in any::<Limb>().prop_filter("divisor must be non-zero", |value| *value != 0),
    ) {
        // SAFETY: the strategy filters out zero divisors.
        let (rem_hi, expected) = unsafe { reference_divrem(limb, rem_seed, divisor) };

        // SAFETY: divisor is nonzero and reference_divrem proves rem_hi < divisor.
        let actual = unsafe { half_limb_test::divrem_1_unchecked(limb, rem_hi, divisor) };
        prop_assert_eq!(actual, expected);
    }
}

/// Computes the exact `DoubleLimb` reference for a valid single-limb divisor.
///
/// # Safety
/// `divisor` must be nonzero.
unsafe fn reference_divrem(limb: Limb, rem_seed: Limb, divisor: Limb) -> (Limb, (Limb, Limb)) {
    // SAFETY: the caller guarantees divisor is nonzero, so checked_rem is Some.
    let rem_hi = unsafe { rem_seed.checked_rem(divisor).unwrap_unchecked() };
    let numerator = (rem_hi as DoubleLimb).wrapping_shl(Limb::BITS) | limb as DoubleLimb;
    let divisor_wide = divisor as DoubleLimb;
    // SAFETY: widening a nonzero Limb preserves nonzeroness.
    let quotient_wide = unsafe { numerator.checked_div(divisor_wide).unwrap_unchecked() };
    // rem_hi < divisor proves the quotient fits exactly in one Limb.
    let quotient = quotient_wide as Limb;
    let remainder =
        numerator.wrapping_sub((quotient as DoubleLimb).wrapping_mul(divisor_wide)) as Limb;
    (rem_hi, (quotient, remainder))
}
