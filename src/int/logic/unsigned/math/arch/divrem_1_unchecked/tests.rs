//! Reference properties for two-limb-by-one-limb division kernels.

#![expect(
    unsafe_code,
    clippy::as_conversions,
    reason = "The properties exercise unsafe hardware contracts; Limb-to-DoubleLimb is widening and quotient/remainder truncation is proven by rem_hi < divisor"
)]
#![cfg_attr(
    target_pointer_width = "32",
    expect(
        clippy::cast_possible_truncation,
        reason = "The properties exercise unsafe hardware contracts; Limb-to-DoubleLimb is widening and quotient/remainder truncation is proven by rem_hi < divisor"
    )
)]

use proptest::prelude::*;

use crate::int::{
    logic::unsigned::math::arch::ArchKernels,
    types::{DoubleLimb, Limb},
};

#[cfg(any(
    all(target_arch = "x86_64", target_pointer_width = "64"),
    all(target_arch = "x86", target_pointer_width = "32"),
    all(target_arch = "s390x", target_pointer_width = "64")
))]
#[path = "half_limb.rs"]
mod half_limb_test;

#[test]
fn normalized_division_covers_half_digit_boundaries() {
    const HALF_BITS: u32 = Limb::BITS >> 1;
    const HALF_BASE: Limb = 1 << HALF_BITS;
    const HIGH_DIGITS: [Limb; 3] = [HALF_BASE >> 1, (HALF_BASE >> 1) + 1, HALF_BASE - 1];
    const LOW_DIGITS: [Limb; 5] = [0, 1, HALF_BASE >> 1, HALF_BASE - 2, HALF_BASE - 1];
    const LOW_LIMBS: [Limb; 6] = [0, 1, HALF_BASE - 1, HALF_BASE, HALF_BASE + 1, Limb::MAX];
    for shift in (0..Limb::BITS).filter(|count| {
        !cfg!(miri) || [0, 1, HALF_BITS - 1, HALF_BITS, Limb::BITS - 1].contains(count)
    }) {
        for high_digit in HIGH_DIGITS {
            for low_digit in LOW_DIGITS {
                let divisor = ((high_digit << HALF_BITS) | low_digit) >> shift;
                // The normalized high bit survives every shift below Limb::BITS.
                for rem_hi in [
                    0,
                    1.min(divisor.saturating_sub(1)),
                    divisor.saturating_sub(1),
                    divisor.saturating_sub(2),
                ] {
                    for limb in LOW_LIMBS {
                        // SAFETY: divisor>0 and each generated rem_hi<divisor.
                        let (_, expected) = unsafe { reference_divrem(limb, rem_hi, divisor) };
                        // SAFETY: the same positive divisor and high remainder
                        // satisfy the selected two-limb division contract.
                        let actual =
                            unsafe { ArchKernels::divrem_1_unchecked(limb, rem_hi, divisor) };
                        assert_eq!(actual, expected);
                        #[cfg(any(
                            all(target_arch = "x86_64", target_pointer_width = "64"),
                            all(target_arch = "x86", target_pointer_width = "32"),
                            all(target_arch = "s390x", target_pointer_width = "64")
                        ))]
                        {
                            // SAFETY: divisor>0 and rem_hi<divisor; this direct
                            // call exercises the shared half-limb core too.
                            let half = unsafe {
                                half_limb_test::divrem_1_unchecked(limb, rem_hi, divisor)
                            };
                            assert_eq!(half, expected);
                        }
                    }
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]
    #[test]
    fn prop_selected_divrem_1_matches_double_limb(
        limb in any::<Limb>(),
        rem_seed in prop_oneof![Just(0), any::<Limb>()],
        divisor in any::<Limb>().prop_filter("divisor must be non-zero", |value| *value != 0),
    ) {
        // SAFETY: the strategy filters out zero divisors.
        let (rem_hi, expected) = unsafe { reference_divrem(limb, rem_seed, divisor) };

        // SAFETY: the generated divisor is non-zero and the modulo construction
        // proves rem_hi < divisor.
        let actual = unsafe { ArchKernels::divrem_1_unchecked(limb, rem_hi, divisor) };
        prop_assert_eq!(actual, expected);
        #[cfg(any(
            all(target_arch = "x86_64", target_pointer_width = "64"),
            all(target_arch = "x86", target_pointer_width = "32"),
            all(target_arch = "s390x", target_pointer_width = "64")
        ))]
        {
            // SAFETY: divisor is nonzero and reference_divrem proves rem_hi < divisor.
            let half = unsafe { half_limb_test::divrem_1_unchecked(limb, rem_hi, divisor) };
            prop_assert_eq!(half, expected);
        }
    }
}

/// Computes the exact `DoubleLimb` reference for a valid single-limb divisor.
///
/// # Safety
/// `divisor` must be nonzero.
unsafe fn reference_divrem(limb: Limb, rem_seed: Limb, divisor: Limb) -> (Limb, (Limb, Limb)) {
    // SAFETY: the caller guarantees divisor is nonzero, so checked_rem is Some.
    let rem_hi = unsafe { rem_seed.checked_rem(divisor).unwrap_unchecked() };
    let numerator = ((rem_hi as DoubleLimb) << Limb::BITS) | limb as DoubleLimb;
    let divisor_wide = divisor as DoubleLimb;
    // SAFETY: widening a nonzero Limb preserves nonzeroness.
    let quotient_wide = unsafe { numerator.checked_div(divisor_wide).unwrap_unchecked() };
    // rem_hi < divisor proves the quotient fits exactly in one Limb.
    let quotient = quotient_wide as Limb;
    // SAFETY: quotient * divisor <= numerator < B^2; the residual is below
    // the one-limb divisor, so multiplication, subtraction and narrowing are exact.
    let remainder = unsafe {
        numerator.unchecked_sub((quotient as DoubleLimb).unchecked_mul(divisor_wide)) as Limb
    };
    (rem_hi, (quotient, remainder))
}
