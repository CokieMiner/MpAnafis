//! Native GCD, quotient estimation, scalar admission, and two-adic factors.

use core::num::NonZeroUsize;

use proptest::prelude::*;

use crate::int::{DoubleLimb, logic::unsigned::math::gcd::binary::lshift_2};

use super::{Gcd, InternalMpUint, LIMB_BITS, Limb};

proptest! {
    #[test]
    fn specialized_binary_leaves_match_euclidean_reference(
        u0 in any::<Limb>(), u1 in any::<Limb>(),
        v0 in any::<Limb>(), v1 in any::<Limb>(),
        shift in 0_u32..Limb::BITS.checked_mul(2).expect("two limb widths fit u32"),
    ) {
        let expected = euclidean_two_limb_reference([u0, u1], [v0, v1]);
        let actual = Gcd::gcd_2([u0, u1], [v0, v1]);
        prop_assert_eq!(actual, expected);
        let scalar_expected = euclidean_two_limb_reference([u0, 0], [v0, 0]);
        prop_assert_eq!(Gcd::gcd_1(u0, v0), scalar_expected[0]);
        let mut shifted_u = [u0, u1];
        let mut shifted_v = [v0, v1];
        lshift_2(&mut shifted_u, shift);
        lshift_2(&mut shifted_v, shift);
        prop_assert_eq!(Gcd::gcd_2(shifted_u, shifted_v), euclidean_two_limb_reference(shifted_u, shifted_v));
    }
}

proptest! {
    #[test]
    fn odd_scalar_cancellation_matches_primitive_remainder(
        limbs in proptest::collection::vec(any::<Limb>(), 2..=129),
        scalar in any::<Limb>(),
    ) {
        let divisor = scalar | 1;
        let wide_divisor = DoubleLimb::try_from(divisor).expect("limb fits double width");
        let wide_remainder = limbs.iter().rev().fold(DoubleLimb::from(0_u8), |remainder, &digit| {
            let column = (remainder << Limb::BITS)
                | DoubleLimb::try_from(digit).expect("limb fits double width");
            column.checked_rem(wide_divisor).expect("positive odd divisor")
        });
        let remainder = Limb::try_from(wide_remainder).expect("remainder is below the scalar divisor");
        let expected = euclidean_two_limb_reference([remainder, 0], [divisor, 0]);
        let denominator = NonZeroUsize::new(divisor).expect("positive odd divisor");
        prop_assert_eq!(Gcd::gcd_odd_limb(&limbs, denominator), expected[0]);
    }

    #[test]
    fn scalar_reduction_matches_primitive_euclid_in_both_orders(
        low in any::<Limb>(),
        high in any::<Limb>(),
        scalar in any::<Limb>(),
    ) {
        let limbs = euclidean_two_limb_reference([low, high], [scalar, 0]);
        let expected = InternalMpUint::from_limbs_2(limbs[0], limbs[1]);
        let large = InternalMpUint::from_limbs_2(low, high);
        let small = InternalMpUint::from_limb(scalar);
        prop_assert_eq!(&large.gcd(&small), &expected);
        prop_assert_eq!(&small.gcd(&large), &expected);
        prop_assert_eq!(Gcd::small_gcd(&large, &small), Some(expected.clone()));
        prop_assert_eq!(Gcd::small_gcd(&small, &large), Some(expected));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 8_192 }))]
    #[test]
    fn hardware_wide_division_matches_primitive(
        numerator in any::<DoubleLimb>(),
        divisor in any::<DoubleLimb>().prop_filter("nonzero divisor", |value| *value != 0),
    ) {
        let expected_quotient = numerator.checked_div(divisor).expect("nonzero divisor");
        let expected_remainder = numerator.checked_rem(divisor).expect("nonzero divisor");
        let (quotient, remainder) = Gcd::div_rem_wide(numerator, divisor);
        let largest_limb = DoubleLimb::try_from(Limb::MAX).expect("limb fits double width");
        if expected_quotient <= largest_limb {
            prop_assert_eq!((quotient, remainder), (expected_quotient, expected_remainder));
        } else {
            prop_assert!(quotient > largest_limb);
        }
    }
}

#[test]
fn wide_quotient_head_estimates_cover_correction_boundaries() {
    for divisor_high in [1_usize, 2, 3, Limb::MAX >> 1, Limb::MAX] {
        for numerator_high in [
            0,
            divisor_high.checked_sub(1).expect("positive high divisor"),
            divisor_high,
            divisor_high.saturating_add(1),
            divisor_high.saturating_mul(divisor_high),
            Limb::MAX,
        ] {
            for divisor_low in [0_usize, 1, Limb::MAX] {
                for numerator_low in [0_usize, 1, Limb::MAX] {
                    let numerator = (DoubleLimb::try_from(numerator_high)
                        .expect("limb fits double width")
                        << LIMB_BITS)
                        | DoubleLimb::try_from(numerator_low).expect("limb fits double width");
                    let divisor = (DoubleLimb::try_from(divisor_high)
                        .expect("limb fits double width")
                        << LIMB_BITS)
                        | DoubleLimb::try_from(divisor_low).expect("limb fits double width");
                    assert_eq!(
                        Gcd::div_rem_wide(numerator, divisor),
                        (
                            numerator.checked_div(divisor).expect("nonzero divisor"),
                            numerator.checked_rem(divisor).expect("nonzero divisor"),
                        )
                    );
                }
            }
        }
    }
}

/// Primitive Euclid is independent of binary GCD and quotient simulation.
pub fn euclidean_two_limb_reference(left: [Limb; 2], right: [Limb; 2]) -> [Limb; 2] {
    let mut first = (DoubleLimb::try_from(left[1]).expect("limb fits double width") << LIMB_BITS)
        | DoubleLimb::try_from(left[0]).expect("limb fits double width");
    let mut second = (DoubleLimb::try_from(right[1]).expect("limb fits double width") << LIMB_BITS)
        | DoubleLimb::try_from(right[0]).expect("limb fits double width");
    while second != 0 {
        let remainder = first
            .checked_rem(second)
            .expect("nonzero Euclidean divisor");
        first = second;
        second = remainder;
    }
    let mask = DoubleLimb::try_from(Limb::MAX).expect("limb fits double width");
    [
        Limb::try_from(first & mask).expect("masked low limb"),
        Limb::try_from(first >> LIMB_BITS).expect("high half is one limb"),
    ]
}

#[test]
fn scalar_powers_of_two_preserve_the_common_valuation() {
    for shift in 0..Limb::BITS {
        let scalar = 1_usize.checked_shl(shift).expect("native shift");
        for low in [0, 1, Limb::MAX, scalar] {
            for high in [0, 1, Limb::MAX] {
                let limbs = euclidean_two_limb_reference([low, high], [scalar, 0]);
                let expected = InternalMpUint::from_limbs_2(limbs[0], limbs[1]);
                let large = InternalMpUint::from_limbs_2(low, high);
                let small = InternalMpUint::from_limb(scalar);
                assert_eq!(large.gcd(&small), expected);
                assert_eq!(small.gcd(&large), expected);
            }
        }
    }
}
