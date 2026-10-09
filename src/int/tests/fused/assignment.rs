//! Properties for allocation-reusing fused arithmetic APIs.

extern crate std;

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use proptest::prelude::{prop_assert, prop_assert_eq, proptest};

use crate::{
    MpInt, MpUint, Precision,
    int::tests::{
        strategies,
        support::{nz, signed_fits},
    },
};

proptest! {
    #[test]
    fn prop_fused_assignment_matches_exact_unlimited_arithmetic(
        unsigned_left in strategies::uint(16),
        unsigned_right in strategies::uint(16),
        unsigned_destination_seed in strategies::uint(16),
        signed_left in strategies::int(16),
        signed_right in strategies::int(16),
        signed_destination_seed in strategies::int(16),
    ) {
        let mut unsigned_sum = unsigned_destination_seed.clone();
        unsigned_sum.assign_add(&unsigned_left, &unsigned_right);
        prop_assert_eq!(unsigned_sum, &unsigned_left + &unsigned_right);

        let mut unsigned_difference = unsigned_destination_seed.clone();
        let underflow = unsigned_difference.assign_sub(&unsigned_left, &unsigned_right);
        if unsigned_left >= unsigned_right {
            prop_assert!(!underflow);
            prop_assert_eq!(unsigned_difference, &unsigned_left - &unsigned_right);
        } else {
            prop_assert!(underflow);
            prop_assert_eq!(&unsigned_difference, &unsigned_destination_seed);
        }

        let mut signed_sum = signed_destination_seed.clone();
        signed_sum.assign_add(&signed_left, &signed_right);
        prop_assert_eq!(signed_sum, &signed_left + &signed_right);

        let mut signed_difference = signed_destination_seed.clone();
        signed_difference.assign_sub(&signed_left, &signed_right);
        prop_assert_eq!(signed_difference, &signed_left - &signed_right);
        let mut unsigned_product = unsigned_destination_seed.clone();
        unsigned_product.assign_mul(&unsigned_left, &unsigned_right);
        prop_assert_eq!(unsigned_product, &unsigned_left * &unsigned_right);

        let mut unsigned_square = unsigned_destination_seed.clone();
        unsigned_square.assign_square(&unsigned_left);
        prop_assert_eq!(unsigned_square, &unsigned_left * &unsigned_left);

        let mut signed_product = signed_destination_seed.clone();
        signed_product.assign_mul(&signed_left, &signed_right);
        prop_assert_eq!(signed_product, &signed_left * &signed_right);

        let mut signed_square = signed_destination_seed.clone();
        signed_square.assign_square(&signed_left);
        prop_assert_eq!(signed_square, &signed_left * &signed_left);
        let mut unsigned_aliased = unsigned_destination_seed;
        unsigned_aliased.assign_mul(&unsigned_left, &unsigned_left);
        prop_assert_eq!(unsigned_aliased, &unsigned_left * &unsigned_left);
        let mut signed_aliased = signed_destination_seed;
        signed_aliased.assign_mul(&signed_left, &signed_left);
        prop_assert_eq!(&signed_aliased, &(&signed_left * &signed_left));
        prop_assert!(!signed_aliased.is_negative());
    }
}

proptest! {
    #[test]
    fn prop_uint_fused_arithmetic_obeys_bounded_contracts(
        bits in 1_usize..=64,
        left_seed in strategies::bounded_uint_wrapped(64),
        right_seed in strategies::bounded_uint_wrapped(64),
        addend_seed in strategies::bounded_uint_wrapped(64),
    ) {
        let width = nz(bits);
        let left = MpUint::with_precision_wrapping(left_seed, width);
        let right = MpUint::with_precision_wrapping(right_seed, width);
        let addend = MpUint::with_precision_wrapping(addend_seed, width);
        let exact_left = MpUint::zero() + &left;
        let exact_right = MpUint::zero() + &right;
        let exact_addend = MpUint::zero() + &addend;

        let exact_sum = &exact_left + &exact_right;
        let mut sum_destination = addend.clone();
        let sum_outcome = catch_unwind(AssertUnwindSafe(|| {
            sum_destination.assign_add(&left, &right);
        }));
        if exact_sum.significant_bits() <= bits {
            prop_assert!(sum_outcome.is_ok(), "representable fused sum must succeed");
            prop_assert_eq!(&sum_destination, &exact_sum);
            prop_assert_eq!(sum_destination.precision(), Precision::Bounded(width));
        } else {
            prop_assert!(sum_outcome.is_err(), "overflowing fused sum must panic");
            prop_assert_eq!(&sum_destination, &addend, "caught panic preserves the destination");
            prop_assert_eq!(sum_destination.precision(), Precision::Bounded(width));
            prop_assert!(sum_destination.checked_add(&MpUint::zero()).is_some(), "receiver remains usable");
        }

        let mut difference_destination = addend.clone();
        let underflow = difference_destination.assign_sub(&left, &right);
        if left >= right {
            let exact_difference = &exact_left - &exact_right;
            prop_assert!(!underflow);
            prop_assert_eq!(&difference_destination, &exact_difference);
        } else {
            prop_assert!(underflow);
            prop_assert_eq!(&difference_destination, &addend);
        }
        prop_assert_eq!(difference_destination.precision(), Precision::Bounded(width));

        let exact_fused = (&exact_left * &exact_right) + &exact_addend;
        let fused_outcome = catch_unwind(AssertUnwindSafe(|| left.mul_add(&right, &addend)));
        if exact_fused.significant_bits() <= bits {
            let fused = fused_outcome.expect("representable fused result must succeed");
            prop_assert_eq!(&fused, &exact_fused);
            prop_assert_eq!(fused.precision(), Precision::Bounded(width));
        } else {
            prop_assert!(fused_outcome.is_err(), "overflowing fused result must panic");
        }
    }
}

proptest! {
    #[test]
    fn prop_int_fused_arithmetic_obeys_bounded_contracts(
        bits in 1_usize..=64,
        left_seed in strategies::bounded_int_wrapped(64),
        right_seed in strategies::bounded_int_wrapped(64),
        addend_seed in strategies::bounded_int_wrapped(64),
    ) {
        let width = nz(bits);
        let left = MpInt::with_precision_wrapping(left_seed, width);
        let right = MpInt::with_precision_wrapping(right_seed, width);
        let addend = MpInt::with_precision_wrapping(addend_seed, width);
        let exact_left = MpInt::zero() + &left;
        let exact_right = MpInt::zero() + &right;
        let exact_addend = MpInt::zero() + &addend;

        let exact_sum = &exact_left + &exact_right;
        let mut sum_destination = addend.clone();
        let sum_outcome = catch_unwind(AssertUnwindSafe(|| {
            sum_destination.assign_add(&left, &right);
        }));
        if signed_fits(&exact_sum, bits) {
            prop_assert!(sum_outcome.is_ok(), "representable fused sum must succeed");
            prop_assert_eq!(&sum_destination, &exact_sum);
            prop_assert_eq!(sum_destination.precision(), Precision::Bounded(width));
        } else {
            prop_assert!(sum_outcome.is_err(), "overflowing fused sum must panic");
            prop_assert_eq!(&sum_destination, &addend, "caught panic preserves the destination");
            prop_assert_eq!(sum_destination.precision(), Precision::Bounded(width));
            prop_assert!(sum_destination.checked_add(&MpInt::zero()).is_some(), "receiver remains usable");
        }

        let exact_difference = &exact_left - &exact_right;
        let mut difference_destination = addend.clone();
        let difference_outcome = catch_unwind(AssertUnwindSafe(|| {
            difference_destination.assign_sub(&left, &right);
        }));
        if signed_fits(&exact_difference, bits) {
            prop_assert!(
                difference_outcome.is_ok(),
                "representable fused difference must succeed"
            );
            prop_assert_eq!(&difference_destination, &exact_difference);
            prop_assert_eq!(difference_destination.precision(), Precision::Bounded(width));
        } else {
            prop_assert!(
                difference_outcome.is_err(),
                "overflowing fused difference must panic"
            );
            prop_assert_eq!(&difference_destination, &addend, "caught panic preserves the destination");
            prop_assert_eq!(difference_destination.precision(), Precision::Bounded(width));
            prop_assert!(
                difference_destination.checked_add(&MpInt::zero()).is_some(),
                "receiver remains usable"
            );
        }

        let exact_fused = (&exact_left * &exact_right) + &exact_addend;
        let fused_outcome = catch_unwind(AssertUnwindSafe(|| left.mul_add(&right, &addend)));
        if signed_fits(&exact_fused, bits) {
            let fused = fused_outcome.expect("representable fused result must succeed");
            prop_assert_eq!(&fused, &exact_fused);
            prop_assert_eq!(fused.precision(), Precision::Bounded(width));
        } else {
            prop_assert!(fused_outcome.is_err(), "overflowing fused result must panic");
        }
    }
}

/// Zero and one products overwrite a destination with a longer initial value.
#[test]
fn fused_product_short_circuits_leave_no_stale_limbs() {
    let seed = (MpUint::one() << 1024_usize) - MpUint::one();
    let zero = MpUint::zero();
    let one = MpUint::one();
    let value = MpUint::from(1_234_567_u32);

    for (label, left, right, expected) in [
        ("zero * value", &zero, &value, &zero),
        ("value * zero", &value, &zero, &zero),
        ("one * value", &one, &value, &value),
        ("value * one", &value, &one, &value),
        ("zero * zero", &zero, &zero, &zero),
        ("one * one", &one, &one, &one),
    ] {
        let mut destination = seed.clone();
        destination.assign_mul(left, right);
        assert_eq!(&destination, expected, "{label}");
    }

    let mut zero_square = seed.clone();
    zero_square.assign_square(&zero);
    assert_eq!(zero_square, zero, "square of zero");

    let mut one_square = seed;
    one_square.assign_square(&one);
    assert_eq!(one_square, one, "square of one");
}
