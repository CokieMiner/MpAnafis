//! GCD, LCM, and Bezout identities with bounded result validation.

extern crate std;

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use proptest::prelude::{ProptestConfig, Strategy, any, prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint, Precision};

use super::{
    strategies,
    support::{exact_limb_vec, nz, signed_fits, uint_from_words},
};

proptest! {
    #[test]
    fn gcd_lcm_and_bezout_obey_arithmetic_identities(
        a in strategies::uint(12), b in strategies::uint(12),
        signed_a in strategies::int(32), signed_b in strategies::int(32),
    ) {
        let unsigned_gcd = a.gcd(&b);
        prop_assert_eq!(&unsigned_gcd, &b.gcd(&a));
        prop_assert_eq!(&a.gcd(&MpUint::zero()), &a);
        prop_assert_eq!(&MpUint::zero().gcd(&a), &a);
        prop_assert_eq!(MpUint::zero().gcd(&MpUint::zero()), MpUint::zero());
        if unsigned_gcd.is_zero() {
            prop_assert!(a.is_zero() && b.is_zero());
        } else {
            prop_assert!((&a % &unsigned_gcd).is_zero());
            prop_assert!((&b % &unsigned_gcd).is_zero());
        }
        let (paired_gcd, lcm) = a.gcd_lcm(&b).expect("unlimited result");
        prop_assert_eq!(&paired_gcd, &unsigned_gcd);
        prop_assert_eq!(a.lcm(&b), Some(lcm.clone()));
        prop_assert_eq!(&unsigned_gcd * &lcm, &a * &b);
        prop_assert_eq!(a.is_coprime(&b), unsigned_gcd.is_one());
        prop_assert_eq!(a.abs_diff(&b), if a >= b { &a - &b } else { &b - &a });
        if !b.is_zero() {
            let (extended_gcd, first, _) = a.extended_gcd(&b).expect("nonzero second operand");
            prop_assert_eq!(&extended_gcd, &unsigned_gcd);
            prop_assert_eq!((&a * &first) % &b, &unsigned_gcd % &b);
        }

        let (signed_gcd, signed_lcm) = signed_a.gcd_lcm(&signed_b).expect("unlimited result");
        prop_assert_eq!(&signed_a.gcd(&signed_b), &signed_gcd);
        prop_assert_eq!(signed_a.lcm(&signed_b), Some(signed_lcm.clone()));
        prop_assert_eq!(&signed_gcd * &signed_lcm, (&signed_a * &signed_b).abs());
        prop_assert_eq!(MpInt::from(signed_a.abs_diff(&signed_b)), (&signed_a - &signed_b).abs());
        if let Some((g, x, y)) = signed_a.extended_gcd(&signed_b) {
            prop_assert_eq!(&g, &signed_gcd);
            prop_assert_eq!(&signed_a * &x + &signed_b * &y, g);
            prop_assert!(!x.is_zero() || !x.is_negative());
            prop_assert!(!y.is_zero() || !y.is_negative());
        }
    }

    #[test]
    fn bounded_gcd_lcm_match_exact_values_and_fit(
        bits in 1_usize..=64, unsigned_a in any::<u64>(), unsigned_b in any::<u64>(),
        signed_a in any::<i64>(), signed_b in any::<i64>(),
    ) {
        let width = nz(bits);
        let unsigned_left = MpUint::with_precision_wrapping(unsigned_a, width);
        let unsigned_right = MpUint::with_precision_wrapping(unsigned_b, width);
        let exact_unsigned_left = MpUint::zero() + &unsigned_left;
        let exact_unsigned_right = MpUint::zero() + &unsigned_right;
        let (exact_unsigned_gcd, exact_unsigned_lcm) = exact_unsigned_left.gcd_lcm(&exact_unsigned_right).expect("unlimited result");
        let fits = exact_unsigned_lcm.significant_bits() <= bits;
        prop_assert_eq!(unsigned_left.lcm(&unsigned_right).is_some(), fits);
        let unsigned_pair = unsigned_left.gcd_lcm(&unsigned_right);
        prop_assert_eq!(unsigned_pair.is_some(), fits);
        if let Some((g, l)) = unsigned_pair {
            prop_assert_eq!(&g, &exact_unsigned_gcd);
            prop_assert_eq!(&l, &exact_unsigned_lcm);
            prop_assert_eq!(g.precision(), Precision::Bounded(width));
            prop_assert_eq!(l.precision(), Precision::Bounded(width));
            prop_assert_eq!(unsigned_left.lcm(&unsigned_right), Some(l));
            prop_assert_eq!((MpUint::zero() + g) * &exact_unsigned_lcm, &exact_unsigned_left * &exact_unsigned_right);
        }

        let signed_left = MpInt::with_precision_wrapping(signed_a, width);
        let signed_right = MpInt::with_precision_wrapping(signed_b, width);
        let exact_signed_left = MpInt::zero() + &signed_left;
        let exact_signed_right = MpInt::zero() + &signed_right;
        let (exact_signed_gcd, exact_signed_lcm) = exact_signed_left.gcd_lcm(&exact_signed_right).expect("unlimited result");
        let gcd_fits = signed_fits(&exact_signed_gcd, bits);
        let lcm_fits = signed_fits(&exact_signed_lcm, bits);
        let gcd = catch_unwind(AssertUnwindSafe(|| signed_left.gcd(&signed_right)));
        prop_assert_eq!(gcd.is_ok(), gcd_fits);
        if let Ok(g) = gcd {
            prop_assert_eq!(&g, &exact_signed_gcd);
            prop_assert_eq!(g.precision(), Precision::Bounded(width));
        }
        let lcm = signed_left.lcm(&signed_right);
        prop_assert_eq!(lcm.is_some(), lcm_fits);
        if let Some(l) = lcm {
            prop_assert_eq!(&l, &exact_signed_lcm);
            prop_assert_eq!(l.precision(), Precision::Bounded(width));
        }
        let signed_pair = signed_left.gcd_lcm(&signed_right);
        prop_assert_eq!(signed_pair.is_some(), gcd_fits && lcm_fits);
        if let Some((g, l)) = signed_pair {
            prop_assert_eq!(&g, &exact_signed_gcd);
            prop_assert_eq!(&l, &exact_signed_lcm);
            prop_assert_eq!(g.precision(), Precision::Bounded(width));
            prop_assert_eq!(l.precision(), Precision::Bounded(width));
            prop_assert_eq!((MpInt::zero() + g) * (MpInt::zero() + l), (&exact_signed_left * &exact_signed_right).abs());
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn wide_gcd_matches_euclidean_remainders(
        (left_words, right_words) in (32_usize..=40).prop_flat_map(|count| (exact_limb_vec(count), exact_limb_vec(count))),
    ) {
        let left = uint_from_words(&left_words);
        let right = uint_from_words(&right_words);
        let actual = left.gcd(&right);
        let mut expected = left;
        let mut divisor = right;
        while !divisor.is_zero() {
            let remainder = &expected % &divisor;
            expected = divisor;
            divisor = remainder;
        }
        prop_assert_eq!(actual, expected);
    }
}

#[test]
fn unit_coefficients_and_signed_minimum_require_valid_result_widths() {
    for (left, right, gcd, first, second) in [
        (0_u16, 1_u16, 1_u16, 0_u16, 1_u16),
        (1, 1, 1, 1, 0),
        (1, 17, 1, 1, 0),
        (17, 1, 1, 0, 1),
        (17, 17, 17, 1, 0),
        (34, 17, 17, 0, 1),
    ] {
        assert_eq!(
            MpUint::from(left).extended_gcd(&MpUint::from(right)),
            Some((MpUint::from(gcd), MpUint::from(first), MpUint::from(second)))
        );
        for first_sign in [-1_i64, 1] {
            for second_sign in [-1_i64, 1] {
                let signed_left = MpInt::from(i64::from(left) * first_sign);
                let signed_right = MpInt::from(i64::from(right) * second_sign);
                let (signed_gcd, left_coefficient, right_coefficient) = signed_left
                    .extended_gcd(&signed_right)
                    .expect("nonzero second operand");
                assert_eq!(signed_gcd, MpInt::from(gcd));
                assert_eq!(
                    &signed_left * &left_coefficient + &signed_right * &right_coefficient,
                    signed_gcd
                );
                assert!(!left_coefficient.is_negative() || !left_coefficient.is_zero());
                assert!(!right_coefficient.is_negative() || !right_coefficient.is_zero());
            }
        }
    }
    for bits in [1, 2, 8, 64, 65, 256] {
        let minimum = MpInt::min_for_precision(bits);
        let zero = MpInt::zero_with_precision(nz(bits));
        assert!(catch_unwind(AssertUnwindSafe(|| minimum.gcd(&zero))).is_err());
        assert!(minimum.gcd_lcm(&zero).is_none());
        assert!(minimum.extended_gcd(&zero).is_none());
    }
}
