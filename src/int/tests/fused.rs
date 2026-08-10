//! Properties for allocation-reusing fused arithmetic APIs.

use alloc::vec;
use core::panic::AssertUnwindSafe;

use super::{std::panic::catch_unwind, *};

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

        let mut unsigned_difference = unsigned_destination_seed;
        let underflow = unsigned_difference.assign_sub(&unsigned_left, &unsigned_right);
        if unsigned_left >= unsigned_right {
            prop_assert!(!underflow);
            prop_assert_eq!(unsigned_difference, &unsigned_left - &unsigned_right);
        } else {
            prop_assert!(underflow);
            prop_assert_eq!(unsigned_difference, ArbiUint::zero());
        }

        let mut signed_sum = signed_destination_seed.clone();
        signed_sum.assign_add(&signed_left, &signed_right);
        prop_assert_eq!(signed_sum, &signed_left + &signed_right);

        let mut signed_difference = signed_destination_seed;
        signed_difference.assign_sub(&signed_left, &signed_right);
        prop_assert_eq!(signed_difference, &signed_left - &signed_right);
    }

    /// The fused product agrees with the operator, for both signednesses.
    ///
    /// The destination seed matters as much as the operands: a fused
    /// assignment writes into a buffer that already holds a value, so a kernel
    /// that fails to overwrite every limb, or that normalises against stale
    /// length, is only caught when the previous contents differ from the
    /// result. Seeding it from an independent generator is what makes that
    /// visible.
    #[test]
    fn prop_fused_product_matches_the_multiplication_operator(
        unsigned_left in strategies::uint(16),
        unsigned_right in strategies::uint(16),
        unsigned_destination_seed in strategies::uint(16),
        signed_left in strategies::int(16),
        signed_right in strategies::int(16),
        signed_destination_seed in strategies::int(16),
    ) {
        let mut unsigned_product = unsigned_destination_seed.clone();
        unsigned_product.assign_mul(&unsigned_left, &unsigned_right);
        prop_assert_eq!(unsigned_product, &unsigned_left * &unsigned_right);

        let mut unsigned_square = unsigned_destination_seed;
        unsigned_square.assign_square(&unsigned_left);
        prop_assert_eq!(unsigned_square, &unsigned_left * &unsigned_left);

        let mut signed_product = signed_destination_seed.clone();
        signed_product.assign_mul(&signed_left, &signed_right);
        prop_assert_eq!(signed_product, &signed_left * &signed_right);

        let mut signed_square = signed_destination_seed;
        signed_square.assign_square(&signed_left);
        prop_assert_eq!(signed_square, &signed_left * &signed_left);
    }

    /// Aliasing the operands takes the squaring tier and still agrees.
    ///
    /// `assign_mul(a, a)` is detected as a square and routed to a different
    /// algorithm from the general product, so it needs its own property rather
    /// than being assumed equivalent to the two-operand case.
    #[test]
    fn prop_fused_product_with_equal_operands_matches_squaring(
        unsigned_value in strategies::uint(16),
        unsigned_destination_seed in strategies::uint(16),
        signed_value in strategies::int(16),
        signed_destination_seed in strategies::int(16),
    ) {
        let mut unsigned_aliased = unsigned_destination_seed.clone();
        unsigned_aliased.assign_mul(&unsigned_value, &unsigned_value);
        let mut unsigned_squared = unsigned_destination_seed;
        unsigned_squared.assign_square(&unsigned_value);
        prop_assert_eq!(unsigned_aliased, unsigned_squared);

        let mut signed_aliased = signed_destination_seed.clone();
        signed_aliased.assign_mul(&signed_value, &signed_value);
        let mut signed_squared = signed_destination_seed;
        signed_squared.assign_square(&signed_value);
        prop_assert_eq!(&signed_aliased, &signed_squared);
        // A square is never negative, including when the operand is.
        prop_assert!(signed_squared >= ArbiInt::zero());
    }
}

/// Zero and one operands, which the fused product short-circuits.
///
/// Each early return writes the destination by a different route -- truncating
/// to zero, or cloning the other operand over whatever was there -- so a stale
/// buffer survives them differently than it survives the general path. The
/// destination seed is deliberately longer than every result here.
#[test]
fn fused_product_short_circuits_leave_no_stale_limbs() {
    let seed = ArbiUint::from(u128::MAX);
    let zero = ArbiUint::zero();
    let one = ArbiUint::one();
    let value = ArbiUint::from(1_234_567_u32);

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

proptest! {
    #[test]
    fn prop_signed_difference_preserves_all_magnitude_limbs(
        limb_count in 2_usize..=12,
        delta in 1_usize..=1024,
    ) {
        // `magnitude = B^limb_count - delta` makes `0 - magnitude` wrap to the
        // short residue `delta` at a subtraction width of `limb_count` limbs.
        // Recovering the magnitude after normalization therefore requires
        // restoring the omitted all-ones upper limbs before canonicalizing.
        let mut limbs = vec![Limb::MAX; limb_count];
        let adjustment = Limb::try_from(delta.wrapping_sub(1))
            .expect("the bounded property delta fits every supported Limb");
        *limbs
            .first_mut()
            .expect("the property always generates at least two limbs") =
            Limb::MAX.wrapping_sub(adjustment);
        let magnitude = ArbiInt::from(ArbiUint {
            value: InternalArbiUint::from_limbs(limbs),
            precision: Precision::Unlimited,
        });
        let zero = ArbiInt::zero();
        let expected_negative = -&magnitude;

        let ordinary_difference = &zero - &magnitude;
        prop_assert_eq!(&ordinary_difference, &expected_negative);
        let ordinary_sum = &zero + &expected_negative;
        prop_assert_eq!(&ordinary_sum, &expected_negative);

        let mut fused_difference = ArbiInt::zero();
        fused_difference.assign_sub(&zero, &magnitude);
        prop_assert_eq!(&fused_difference, &expected_negative);

        let mut fused_sum = ArbiInt::zero();
        fused_sum.assign_add(&zero, &expected_negative);
        prop_assert_eq!(&fused_sum, &expected_negative);

        let mut in_place_difference = ArbiInt::zero();
        in_place_difference -= &magnitude;
        prop_assert_eq!(&in_place_difference, &expected_negative);

        let mut in_place_sum = ArbiInt::zero();
        in_place_sum += &expected_negative;
        prop_assert_eq!(&in_place_sum, &expected_negative);
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
        let left = ArbiUint {
            value: left_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let right = ArbiUint {
            value: right_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let addend = ArbiUint {
            value: addend_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let mut exact_left = left.clone();
        exact_left.precision = Precision::Unlimited;
        let mut exact_right = right.clone();
        exact_right.precision = Precision::Unlimited;
        let mut exact_addend = addend.clone();
        exact_addend.precision = Precision::Unlimited;

        let exact_sum = &exact_left + &exact_right;
        let mut sum_destination = addend.clone();
        let sum_outcome = catch_unwind(AssertUnwindSafe(|| {
            sum_destination.assign_add(&left, &right);
        }));
        if exact_sum.value.required_unsigned_bits_for_bounded_storage() <= bits {
            prop_assert!(sum_outcome.is_ok(), "representable fused sum must succeed");
            prop_assert_eq!(&sum_destination, &exact_sum);
            prop_assert_eq!(sum_destination.precision, Precision::Bounded(width));
        } else {
            prop_assert!(sum_outcome.is_err(), "overflowing fused sum must panic");
            prop_assert!(sum_destination.is_zero(), "caught panic must leave valid zero");
            prop_assert_eq!(sum_destination.precision, Precision::Bounded(width));
            sum_destination.debug_assert_valid();
            prop_assert!(sum_destination.checked_add(&left).is_some(), "receiver remains usable");
        }

        let mut difference_destination = addend.clone();
        let underflow = difference_destination.assign_sub(&left, &right);
        if left >= right {
            let exact_difference = &exact_left - &exact_right;
            prop_assert!(!underflow);
            prop_assert_eq!(&difference_destination, &exact_difference);
        } else {
            prop_assert!(underflow);
            prop_assert!(difference_destination.is_zero());
        }
        prop_assert_eq!(difference_destination.precision, Precision::Bounded(width));
        difference_destination.debug_assert_valid();

        let exact_fused = (&exact_left * &exact_right) + &exact_addend;
        let fused_outcome = catch_unwind(AssertUnwindSafe(|| left.mul_add(&right, &addend)));
        if exact_fused.value.required_unsigned_bits_for_bounded_storage() <= bits {
            let fused = fused_outcome.expect("representable fused result must succeed");
            prop_assert_eq!(&fused, &exact_fused);
            prop_assert_eq!(fused.precision, Precision::Bounded(width));
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
        let left = ArbiInt {
            value: left_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let right = ArbiInt {
            value: right_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let addend = ArbiInt {
            value: addend_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let mut exact_left = left.clone();
        exact_left.precision = Precision::Unlimited;
        let mut exact_right = right.clone();
        exact_right.precision = Precision::Unlimited;
        let mut exact_addend = addend.clone();
        exact_addend.precision = Precision::Unlimited;

        let exact_sum = &exact_left + &exact_right;
        let mut sum_destination = addend.clone();
        let sum_outcome = catch_unwind(AssertUnwindSafe(|| {
            sum_destination.assign_add(&left, &right);
        }));
        if exact_sum.value.required_signed_bits_for_bounded_storage() <= bits {
            prop_assert!(sum_outcome.is_ok(), "representable fused sum must succeed");
            prop_assert_eq!(&sum_destination, &exact_sum);
            prop_assert_eq!(sum_destination.precision, Precision::Bounded(width));
        } else {
            prop_assert!(sum_outcome.is_err(), "overflowing fused sum must panic");
            prop_assert!(sum_destination.is_zero(), "caught panic must leave valid zero");
            prop_assert_eq!(sum_destination.precision, Precision::Bounded(width));
            sum_destination.debug_assert_valid();
            prop_assert!(sum_destination.checked_add(&left).is_some(), "receiver remains usable");
        }

        let exact_difference = &exact_left - &exact_right;
        let mut difference_destination = addend.clone();
        let difference_outcome = catch_unwind(AssertUnwindSafe(|| {
            difference_destination.assign_sub(&left, &right);
        }));
        if exact_difference
            .value
            .required_signed_bits_for_bounded_storage()
            <= bits
        {
            prop_assert!(
                difference_outcome.is_ok(),
                "representable fused difference must succeed"
            );
            prop_assert_eq!(&difference_destination, &exact_difference);
            prop_assert_eq!(difference_destination.precision, Precision::Bounded(width));
        } else {
            prop_assert!(
                difference_outcome.is_err(),
                "overflowing fused difference must panic"
            );
            prop_assert!(
                difference_destination.is_zero(),
                "caught panic must leave valid zero"
            );
            prop_assert_eq!(difference_destination.precision, Precision::Bounded(width));
            difference_destination.debug_assert_valid();
            prop_assert!(
                difference_destination.checked_add(&left).is_some(),
                "receiver remains usable"
            );
        }

        let exact_fused = (&exact_left * &exact_right) + &exact_addend;
        let fused_outcome = catch_unwind(AssertUnwindSafe(|| left.mul_add(&right, &addend)));
        if exact_fused.value.required_signed_bits_for_bounded_storage() <= bits {
            let fused = fused_outcome.expect("representable fused result must succeed");
            prop_assert_eq!(&fused, &exact_fused);
            prop_assert_eq!(fused.precision, Precision::Bounded(width));
        } else {
            prop_assert!(fused_outcome.is_err(), "overflowing fused result must panic");
        }
    }
}

proptest! {
    #[test]
    fn prop_signed_mul_add_checks_only_the_exact_final_result(
        bits in 3_usize..=64,
        selection_seed in any::<u64>(),
    ) {
        let magnitude_bits = bits.wrapping_sub(1);
        let shift = u32::try_from(magnitude_bits).expect("property width fits u32");
        let maximum = 1_i128
            .checked_shl(shift)
            .expect("property width is at most 64")
            .wrapping_sub(1);
        let overflow_floor = maximum.div_euclid(2).wrapping_add(1);
        let selection_span = maximum
            .wrapping_sub(overflow_floor)
            .wrapping_add(1);
        let selected_offset = i128::from(selection_seed).rem_euclid(selection_span);
        let factor_value = overflow_floor.wrapping_add(selected_offset);
        let width = nz(bits);
        let factor = ArbiInt::with_precision_checked(factor_value, width)
            .expect("selected positive factor fits");
        let multiplier = ArbiInt::with_precision_checked(2_i8, width)
            .expect("two fits every tested width");
        let addend = ArbiInt::with_precision_checked(factor_value.wrapping_neg(), width)
            .expect("negative factor fits");

        let product_outcome = catch_unwind(AssertUnwindSafe(|| &factor * &multiplier));
        prop_assert!(product_outcome.is_err(), "intermediate product must overflow");

        let fused = factor.mul_add(&multiplier, &addend);
        prop_assert_eq!(fused, factor, "exact addition brings the final result back in range");
    }
}
