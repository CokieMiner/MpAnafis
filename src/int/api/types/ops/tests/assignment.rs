//! Assignment-operator precision properties.

extern crate std;

use core::{
    ops::{
        AddAssign, BitAndAssign, BitOrAssign, BitXorAssign, DivAssign, MulAssign, RemAssign,
        SubAssign,
    },
    panic::AssertUnwindSafe,
};
use std::panic::catch_unwind;

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{BoundedPrecision, MpInt, MpUint, Precision};

proptest! {
    #[test]
    fn signed_addition_preserves_destination_across_width_proofs(
        bits in 1_usize..=9,
        left_seed in any::<i8>(),
        right_seed in any::<i8>(),
        owned in any::<bool>(),
    ) {
        let limit = 1_i128 << bits.checked_sub(1).expect("nonzero test width");
        let minimum = limit.checked_neg().expect("small signed bound");
        let maximum = limit.checked_sub(1).expect("positive bound");
        let left_native = i128::from(left_seed).clamp(minimum, maximum);
        let exact = left_native.checked_add(i128::from(right_seed)).expect("sum of small values");
        let width = BoundedPrecision::new(bits).expect("positive test width");
        let original = MpInt::with_precision_checked(left_native, width).expect("clamped value fits");
        let mut destination = original.clone();
        destination.reserve(16);
        let capacity = destination.capacity();
        let right = MpInt::from(right_seed);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if owned {
                AddAssign::add_assign(&mut destination, right);
            } else {
                AddAssign::add_assign(&mut destination, &right);
            }
        }));
        if (minimum..=maximum).contains(&exact) {
            prop_assert!(outcome.is_ok());
            prop_assert_eq!(&destination, &MpInt::from(exact));
        } else {
            prop_assert!(outcome.is_err());
            prop_assert_eq!(&destination, &original);
            prop_assert_eq!(destination.capacity(), capacity);
        }
        prop_assert_eq!(destination.precision(), original.precision());
    }

    #[test]
    fn unsigned_assignment_preserves_lhs_precision(
        bits in 8_usize..=64,
        lhs_value in 0_u8..=127,
        rhs_value in 1_u8..=127,
        small_lhs in 0_u8..=15,
        small_rhs in 1_u8..=15,
    ) {
        let width = BoundedPrecision::new(bits).expect("positive test width");
        let precision = Precision::Bounded(width);
        let lhs = MpUint::with_precision_checked(lhs_value, width).expect("seven bits fit");
        let rhs = MpUint::from(rhs_value);

        let mut sum_ref = lhs.clone();
        AddAssign::add_assign(&mut sum_ref, &rhs);
        prop_assert_eq!(sum_ref.precision(), precision);

        let mut sum_owned = lhs.clone();
        AddAssign::add_assign(&mut sum_owned, rhs.clone());
        prop_assert_eq!(sum_owned.precision(), precision);

        let (larger, smaller) = if lhs_value >= rhs_value {
            (lhs_value, rhs_value)
        } else {
            (rhs_value, lhs_value)
        };
        let mut difference = MpUint::with_precision_checked(larger, width).expect("seven bits fit");
        SubAssign::sub_assign(&mut difference, MpUint::from(smaller));
        prop_assert_eq!(difference.precision(), precision);

        let mut product = MpUint::with_precision_checked(small_lhs, width).expect("four bits fit");
        MulAssign::mul_assign(&mut product, MpUint::from(small_rhs));
        prop_assert_eq!(product.precision(), precision);

        let mut quotient = lhs.clone();
        DivAssign::div_assign(&mut quotient, &rhs);
        prop_assert_eq!(quotient.precision(), precision);

        let mut remainder = lhs.clone();
        RemAssign::rem_assign(&mut remainder, rhs.clone());
        prop_assert_eq!(remainder.precision(), precision);

        let mut and_value = lhs.clone();
        BitAndAssign::bitand_assign(&mut and_value, &rhs);
        prop_assert_eq!(and_value.precision(), precision);

        let mut or_value = lhs.clone();
        BitOrAssign::bitor_assign(&mut or_value, rhs.clone());
        prop_assert_eq!(or_value.precision(), precision);

        let mut xor_value = lhs;
        BitXorAssign::bitxor_assign(&mut xor_value, rhs);
        prop_assert_eq!(xor_value.precision(), precision);
    }

    #[test]
    fn signed_assignment_preserves_lhs_precision(
        bits in 8_usize..=64,
        lhs_value in -31_i8..=31,
        rhs_value in 1_i8..=31,
        small_lhs in -10_i8..=10,
        small_rhs in 1_i8..=10,
    ) {
        let width = BoundedPrecision::new(bits).expect("positive test width");
        let precision = Precision::Bounded(width);
        let lhs = MpInt::with_precision_checked(lhs_value, width).expect("six signed bits fit");
        let rhs = MpInt::from(rhs_value);

        let mut sum_ref = lhs.clone();
        AddAssign::add_assign(&mut sum_ref, &rhs);
        prop_assert_eq!(sum_ref.precision(), precision);

        let mut sum_owned = lhs.clone();
        AddAssign::add_assign(&mut sum_owned, rhs.clone());
        prop_assert_eq!(sum_owned.precision(), precision);

        let mut difference = lhs.clone();
        SubAssign::sub_assign(&mut difference, rhs.clone());
        prop_assert_eq!(difference.precision(), precision);

        let mut product = MpInt::with_precision_checked(small_lhs, width).expect("five signed bits fit");
        MulAssign::mul_assign(&mut product, MpInt::from(small_rhs));
        prop_assert_eq!(product.precision(), precision);

        let mut quotient = lhs.clone();
        DivAssign::div_assign(&mut quotient, &rhs);
        prop_assert_eq!(quotient.precision(), precision);

        let mut remainder = lhs.clone();
        RemAssign::rem_assign(&mut remainder, rhs.clone());
        prop_assert_eq!(remainder.precision(), precision);

        let mut and_value = lhs.clone();
        BitAndAssign::bitand_assign(&mut and_value, &rhs);
        prop_assert_eq!(and_value.precision(), precision);

        let mut or_value = lhs.clone();
        BitOrAssign::bitor_assign(&mut or_value, rhs.clone());
        prop_assert_eq!(or_value.precision(), precision);

        let mut xor_value = lhs;
        BitXorAssign::bitxor_assign(&mut xor_value, rhs);
        prop_assert_eq!(xor_value.precision(), precision);
    }
}

#[test]
fn bounded_signed_addition_reuses_heap_storage_when_widths_prove_fit() {
    let limb_bits = usize::try_from(usize::BITS).expect("pointer width fits usize");
    let value_bits = limb_bits.checked_mul(8).expect("small test width");
    let precision_bits = value_bits.checked_add(3).expect("small bounded width");
    let width = BoundedPrecision::new(precision_bits).expect("positive test width");
    let magnitude = MpInt::one() << value_bits;
    for left_negative in [false, true] {
        for right_negative in [false, true] {
            let left = if left_negative {
                -&magnitude
            } else {
                magnitude.clone()
            };
            let half = &magnitude >> 1_usize;
            let right = if right_negative { -half } else { half };
            let expected = &left + &right;
            for owned in [false, true] {
                let mut destination = MpInt::zero_with_precision(width);
                destination.reserve(32);
                destination.assign_add(&left, &MpInt::zero());
                let capacity = destination.capacity();
                if owned {
                    destination += right.clone();
                } else {
                    destination += &right;
                }
                assert_eq!(destination, expected);
                assert_eq!(destination.capacity(), capacity);
                assert_eq!(
                    destination.precision(),
                    Precision::new_bounded(precision_bits).expect("valid width")
                );
            }
        }
    }
}

#[test]
fn signed_addition_validates_the_sum_when_the_right_operand_is_wider() {
    let width = BoundedPrecision::new(8).expect("valid test width");
    for (left, right, expected) in [
        (-128_i16, 255_i16, Some(127_i16)),
        (127, -255, Some(-128)),
        (-128, 256, None),
        (127, -256, None),
    ] {
        for owned in [false, true] {
            let original = MpInt::with_precision_checked(left, width).expect("signed byte fits");
            let mut destination = original.clone();
            let rhs = MpInt::from(right);
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                if owned {
                    destination += rhs;
                } else {
                    destination += &rhs;
                }
            }));
            if let Some(value) = expected {
                assert!(outcome.is_ok());
                assert_eq!(destination, MpInt::from(value));
            } else {
                assert!(outcome.is_err());
                assert_eq!(destination, original);
            }
            assert_eq!(destination.precision(), original.precision());
        }
    }
}
