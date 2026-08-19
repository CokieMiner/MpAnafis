//! Properties for allocation-reusing fused arithmetic APIs.

use alloc::vec;
use core::panic::AssertUnwindSafe;

use crate::int::InternalMpInt;

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
            prop_assert_eq!(unsigned_difference, MpUint::zero());
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
        prop_assert!(signed_squared >= MpInt::zero());
    }
}

fn reconstruct_signed_words(lower: &MpInt, upper: &MpInt, width: usize) -> InternalMpInt {
    let mut bits = upper.value.to_tc_bits(width).shl(width);
    bits.add_assign(&lower.value.to_tc_bits(width));
    InternalMpInt::from_tc_bits(bits, width * 2)
}

#[test]
fn signed_widening_words_reconstruct_negative_and_boundary_products() {
    let width = nz(8);
    for (left_value, right_value) in [
        (-128_i16, 1_i16),
        (-1, 2),
        (-128, -1),
        (127, 127),
        (-128, -128),
    ] {
        let left = MpInt::with_precision_checked(left_value, width).expect("left fits");
        let right = MpInt::with_precision_checked(right_value, width).expect("right fits");
        let expected = MpInt::from(left_value) * MpInt::from(right_value);

        let (lower, upper) = left.widening_mul(&right);
        assert_eq!(
            reconstruct_signed_words(&lower, &upper, width.get()),
            expected.value,
            "signed word reconstruction for {left_value} * {right_value}"
        );
        let (try_lower, try_upper) = left.try_widening_mul(&right).expect("bounded width");
        assert_eq!(try_lower, lower);
        assert_eq!(try_upper, upper);
    }
}

#[test]
fn signed_carrying_words_reconstruct_negative_and_boundary_sums() {
    let width = nz(8);
    let left = MpInt::with_precision_checked(-128_i16, width).expect("left fits");
    let right = MpInt::with_precision_checked(-128_i16, width).expect("right fits");
    let carry = MpInt::with_precision_checked(-3_i16, width).expect("carry fits");
    let expected = MpInt::from(-128_i16) * MpInt::from(-128_i16) + MpInt::from(-3_i16);

    let (lower, upper) = left.carrying_mul(&right, &carry);
    assert_eq!(
        reconstruct_signed_words(&lower, &upper, width.get()),
        expected.value
    );
    let (try_lower, try_upper) = left
        .try_carrying_mul(&right, &carry)
        .expect("bounded width");
    assert_eq!(try_lower, lower);
    assert_eq!(try_upper, upper);

    let carry1 = MpInt::with_precision_checked(127_i16, width).expect("carry1 fits");
    let carry2 = MpInt::with_precision_checked(127_i16, width).expect("carry2 fits");
    let expected_add =
        MpInt::from(-128_i16) * MpInt::from(-128_i16) + MpInt::from(127_i16) + MpInt::from(127_i16);
    let (add_lower, add_upper) = left.carrying_mul_add(&right, &carry1, &carry2);
    assert_eq!(
        reconstruct_signed_words(&add_lower, &add_upper, width.get()),
        expected_add.value
    );
}

#[test]
fn signed_widening_unbounded_returns_exact_lower_and_rejects_try_split() {
    let left = MpInt::from(-129_i16);
    let right = MpInt::from(7_i16);
    let carry = MpInt::from(-3_i16);
    let expected_product = &left * &right;
    let expected_carrying = &expected_product + &carry;

    let (lower, upper) = left.widening_mul(&right);
    assert_eq!(lower, expected_product);
    assert_eq!(upper, MpInt::zero());
    assert_eq!(left.try_widening_mul(&right), Err(MpError::WidthRequired));

    let (carry_lower, carry_upper) = left.carrying_mul(&right, &carry);
    assert_eq!(carry_lower, expected_carrying);
    assert_eq!(carry_upper, MpInt::zero());
    assert_eq!(
        left.try_carrying_mul(&right, &carry),
        Err(MpError::WidthRequired)
    );

    let (add_lower, add_upper) = left.carrying_mul_add(&right, &carry, &MpInt::one());
    assert_eq!(add_lower, &expected_carrying + MpInt::one());
    assert_eq!(add_upper, MpInt::zero());
}

/// Zero and one operands, which the fused product short-circuits.
///
/// Each early return writes the destination by a different route -- truncating
/// to zero, or cloning the other operand over whatever was there -- so a stale
/// buffer survives them differently than it survives the general path. The
/// destination seed is deliberately longer than every result here.
#[test]
fn fused_product_short_circuits_leave_no_stale_limbs() {
    let seed = MpUint::from(u128::MAX);
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
        let magnitude = MpInt::from(MpUint {
            value: InternalMpUint::from_limbs(limbs),
            precision: Precision::Unlimited,
        });
        let zero = MpInt::zero();
        let expected_negative = -&magnitude;

        let ordinary_difference = &zero - &magnitude;
        prop_assert_eq!(&ordinary_difference, &expected_negative);
        let ordinary_sum = &zero + &expected_negative;
        prop_assert_eq!(&ordinary_sum, &expected_negative);

        let mut fused_difference = MpInt::zero();
        fused_difference.assign_sub(&zero, &magnitude);
        prop_assert_eq!(&fused_difference, &expected_negative);

        let mut fused_sum = MpInt::zero();
        fused_sum.assign_add(&zero, &expected_negative);
        prop_assert_eq!(&fused_sum, &expected_negative);

        let mut in_place_difference = MpInt::zero();
        in_place_difference -= &magnitude;
        prop_assert_eq!(&in_place_difference, &expected_negative);

        let mut in_place_sum = MpInt::zero();
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
        let left = MpUint {
            value: left_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let right = MpUint {
            value: right_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let addend = MpUint {
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
        let left = MpInt {
            value: left_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let right = MpInt {
            value: right_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(width),
        };
        let addend = MpInt {
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
        let factor = MpInt::with_precision_checked(factor_value, width)
            .expect("selected positive factor fits");
        let multiplier = MpInt::with_precision_checked(2_i8, width)
            .expect("two fits every tested width");
        let addend = MpInt::with_precision_checked(factor_value.wrapping_neg(), width)
            .expect("negative factor fits");

        let product_outcome = catch_unwind(AssertUnwindSafe(|| &factor * &multiplier));
        prop_assert!(product_outcome.is_err(), "intermediate product must overflow");

        let fused = factor.mul_add(&multiplier, &addend);
        prop_assert_eq!(fused, factor, "exact addition brings the final result back in range");
    }

    #[test]
    fn prop_widening_and_carrying_mul_identities(
        u_a in strategies::uint(8),
        u_b in strategies::uint(8),
        u_c in strategies::uint(8),
        u_d in strategies::uint(8),
        shift in 0_usize..=128,
    ) {
        prop_assert_eq!(u_a.mul_2exp(shift), &u_a << shift);
        prop_assert_eq!(u_a.div_2exp(shift), &u_a >> shift);

        let (lo, _hi) = u_a.widening_mul(&u_b);
        prop_assert_eq!(lo, &u_a * &u_b);

        let (c_lo, _c_hi) = u_a.carrying_mul(&u_b, &u_c);
        let expected_carrying = (&u_a * &u_b) + &u_c;
        prop_assert_eq!(c_lo, expected_carrying);

        let (ca_lo, _ca_hi) = u_a.carrying_mul_add(&u_b, &u_c, &u_d);
        let expected_carrying_add = (&u_a * &u_b) + &u_c + &u_d;
        prop_assert_eq!(ca_lo, expected_carrying_add);
    }

    #[test]
    fn prop_bounded_widening_mul_split(
        bits in 8_usize..=64,
        a_val in any::<u64>(),
        b_val in any::<u64>(),
    ) {
        let width = nz(bits);
        let mask = if bits == 64 { u64::MAX } else { (1_u64 << bits).wrapping_sub(1) };
        let a_bounded = a_val & mask;
        let b_bounded = b_val & mask;
        let u_a = MpUint::with_precision_checked(a_bounded, width).expect("bounded value fits");
        let u_b = MpUint::with_precision_checked(b_bounded, width).expect("bounded value fits");

        let (lo, hi) = u_a.widening_mul(&u_b);
        let (try_lo, try_hi) = u_a.try_widening_mul(&u_b).expect("bounded precision supports try_widening_mul");
        prop_assert_eq!(&lo, &try_lo);
        prop_assert_eq!(&hi, &try_hi);

        let full_prod = u128::from(a_bounded).wrapping_mul(u128::from(b_bounded));
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation, reason = "mask restricts to u64")]
        let expected_lo = (full_prod & u128::from(mask)) as u64;
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation, reason = "bits <= 64 restricts upper to u64")]
        let expected_hi = (full_prod >> bits) as u64;
        prop_assert_eq!(lo, MpUint::with_precision_checked(expected_lo, width).expect("expected lo fits"));
        prop_assert_eq!(hi, MpUint::with_precision_checked(expected_hi, width).expect("expected hi fits"));
    }
}
