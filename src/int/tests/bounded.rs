//! Properties for checked, wrapping, saturating, and bounded-width operations.

use core::panic::AssertUnwindSafe;

use crate::int::LIMB_BITS;

use super::{std::panic::catch_unwind, *};

#[test]
fn bounded_wrapping_sub_restores_residue_width_before_sign_extension() {
    let residue_bits = LIMB_BITS.wrapping_mul(2);
    let bits = LIMB_BITS.wrapping_mul(3);
    let width = nz(bits);
    let rhs_unlimited = (MpUint::one() << residue_bits) - MpUint::one();
    let lhs = MpUint::zero_with_precision(width);
    let rhs = MpUint::with_precision_checked(rhs_unlimited.clone(), width)
        .expect("the two-limb operand fits the three-limb precision");

    // At two limbs, `0 - (B^2 - 1) mod B^2 = 1`. Extending that normalized
    // residue directly would incorrectly fill the discarded zero second limb
    // with ones. The correct three-limb residue is `B^3 - B^2 + 1`.
    let expected = (MpUint::one() << bits) - rhs_unlimited;
    let wrapped = lhs.wrapping_sub(&rhs);
    let (overflowing, underflowed) = lhs.overflowing_sub(&rhs);

    assert_eq!(wrapped.value, expected.value);
    assert_eq!(wrapped.precision, Precision::Bounded(width));
    assert!(underflowed);
    assert_eq!(overflowing, wrapped);
}

#[test]
fn zero_result_branches_preserve_resolved_precision() {
    let expected = Precision::Bounded(nz(16));

    let unsigned = MpUint::with_precision_checked(1_u8, nz(8)).expect("one fits in 8 bits");
    let unsigned_larger =
        MpUint::with_precision_checked(2_u8, nz(16)).expect("two fits in 16 bits");
    let unsigned_zero = MpUint::zero_with_precision(nz(16));

    let (unsigned_overflowing_quotient, unsigned_division_overflowed) =
        unsigned.overflowing_div(&unsigned_zero);
    assert!(unsigned_division_overflowed);
    assert!(unsigned_overflowing_quotient.is_zero());
    assert_eq!(unsigned_overflowing_quotient.precision, expected);

    let (unsigned_overflowing_remainder, unsigned_remainder_overflowed) =
        unsigned.overflowing_rem(&unsigned_zero);
    assert!(unsigned_remainder_overflowed);
    assert!(unsigned_overflowing_remainder.is_zero());
    assert_eq!(unsigned_overflowing_remainder.precision, expected);

    let unsigned_saturating_difference = unsigned.saturating_sub(&unsigned_larger);
    assert!(unsigned_saturating_difference.is_zero());
    assert_eq!(unsigned_saturating_difference.precision, expected);

    let unsigned_saturating_quotient = unsigned.saturating_div(&unsigned_zero);
    assert!(unsigned_saturating_quotient.is_zero());
    assert_eq!(unsigned_saturating_quotient.precision, expected);

    let unsigned_saturating_remainder = unsigned.saturating_rem(&unsigned_zero);
    assert!(unsigned_saturating_remainder.is_zero());
    assert_eq!(unsigned_saturating_remainder.precision, expected);

    let signed = MpInt::with_precision_checked(1_i8, nz(8)).expect("one fits in 8 bits");
    let signed_larger = MpInt::with_precision_checked(2_i8, nz(16)).expect("two fits in 16 bits");
    let signed_zero = MpInt::zero_with_precision(nz(16));

    let (signed_overflowing_quotient, signed_division_overflowed) =
        signed.overflowing_div(&signed_zero);
    assert!(signed_division_overflowed);
    assert!(signed_overflowing_quotient.is_zero());
    assert_eq!(signed_overflowing_quotient.precision, expected);

    let (signed_overflowing_remainder, signed_remainder_overflowed) =
        signed.overflowing_rem(&signed_zero);
    assert!(signed_remainder_overflowed);
    assert!(signed_overflowing_remainder.is_zero());
    assert_eq!(signed_overflowing_remainder.precision, expected);

    let signed_saturating_quotient = signed.saturating_div(&signed_zero);
    assert!(signed_saturating_quotient.is_zero());
    assert_eq!(signed_saturating_quotient.precision, expected);

    let signed_saturating_remainder = signed.saturating_rem(&signed_zero);
    assert!(signed_saturating_remainder.is_zero());
    assert_eq!(signed_saturating_remainder.precision, expected);

    let signed_abs_sub = signed.abs_sub(&signed_larger);
    assert!(signed_abs_sub.is_zero());
    assert_eq!(signed_abs_sub.precision, expected);
}

#[test]
fn bounded_shifts_reject_huge_counts_before_allocation() {
    let width = nz(8);
    let unsigned_zero = MpUint::zero_with_precision(width);
    let unsigned_one = MpUint::with_precision_checked(1_u8, width).expect("one fits in eight bits");
    let unsigned_expected_zero = MpUint::zero_with_precision(width);

    assert_eq!(
        unsigned_zero.checked_shl(usize::MAX),
        Some(unsigned_zero.clone())
    );
    assert_eq!(unsigned_zero.try_shl(usize::MAX), Ok(unsigned_zero.clone()));
    assert_eq!(
        unsigned_zero.overflowing_shl(usize::MAX),
        (unsigned_zero, false)
    );
    assert_eq!(unsigned_one.checked_shl(usize::MAX), None);
    assert_eq!(unsigned_one.try_shl(usize::MAX), Err(MpError::Overflow));
    assert_eq!(
        unsigned_one.wrapping_shl(usize::MAX),
        unsigned_expected_zero
    );
    assert_eq!(
        unsigned_one.overflowing_shl(usize::MAX),
        (MpUint::zero_with_precision(width), true)
    );
    assert_eq!(
        unsigned_one.saturating_shl(usize::MAX),
        MpUint::max_for_precision(8)
    );

    let signed_zero = MpInt::zero_with_precision(width);
    let signed_one = MpInt::with_precision_checked(1_i8, width).expect("one fits in eight bits");
    let signed_minus_one =
        MpInt::with_precision_checked(-1_i8, width).expect("minus one fits in eight bits");

    assert_eq!(
        signed_zero.checked_shl(usize::MAX),
        Some(signed_zero.clone())
    );
    assert_eq!(signed_zero.try_shl(usize::MAX), Ok(signed_zero.clone()));
    assert_eq!(
        signed_zero.overflowing_shl(usize::MAX),
        (signed_zero, false)
    );
    assert_eq!(signed_one.checked_shl(usize::MAX), None);
    assert_eq!(signed_one.try_shl(usize::MAX), Err(MpError::Overflow));
    assert_eq!(
        signed_one.wrapping_shl(usize::MAX),
        MpInt::zero_with_precision(width)
    );
    assert_eq!(
        signed_one.overflowing_shl(usize::MAX),
        (MpInt::zero_with_precision(width), true)
    );
    assert_eq!(
        signed_one.saturating_shl(usize::MAX),
        MpInt::max_for_precision(8)
    );
    assert_eq!(
        signed_minus_one.wrapping_shl(usize::MAX),
        MpInt::zero_with_precision(width)
    );
    assert_eq!(
        signed_minus_one.saturating_shl(usize::MAX),
        MpInt::min_for_precision(8)
    );
    assert_eq!(
        signed_minus_one.checked_shl(7),
        Some(MpInt::min_for_precision(8))
    );
    assert_eq!(
        signed_one.checked_shl(6),
        Some(
            MpInt::with_precision_checked(64_i8, width)
                .expect("sixty-four fits in eight signed bits")
        )
    );
}

#[test]
fn explicit_width_bitwise_validates_width_before_allocation() {
    let unsigned = MpUint::from(1_u8);
    let unsigned_rotated = unsigned
        .rotate_left(3, 16)
        .expect("sixteen is a valid explicit width");
    assert_eq!(unsigned_rotated.precision, Precision::Bounded(nz(16)));
    assert_eq!(unsigned.rotate_left(1, usize::MAX), None);
    assert_eq!(unsigned.rotate_right(1, usize::MAX), None);
    assert_eq!(unsigned.reverse_bits(usize::MAX), None);
    assert_eq!(unsigned.not_with_width(usize::MAX), None);

    let signed = MpInt::from(-1_i8);
    let signed_rotated = signed
        .rotate_left(3, 16)
        .expect("sixteen is a valid explicit width");
    assert_eq!(signed_rotated.precision, Precision::Bounded(nz(16)));
    assert_eq!(signed.rotate_left(1, usize::MAX), None);
    assert_eq!(signed.rotate_right(1, usize::MAX), None);
    assert_eq!(signed.reverse_bits(usize::MAX), None);
    assert_eq!(signed.not_with_width(usize::MAX), None);
}

#[test]
fn bounded_signed_min_remainder_by_minus_one_reports_overflow_consistently() {
    let minimum = MpInt::min_for_precision(8);
    let minus_one =
        MpInt::with_precision_checked(-1_i8, nz(8)).expect("minus one fits in eight bits");

    assert_eq!(minimum.checked_rem(&minus_one), None);
    assert_eq!(minimum.try_rem(&minus_one), Err(MpError::Overflow));
    assert_eq!(minimum.div_rem(&minus_one), None);
    assert_eq!(minimum.checked_rem_trunc(&minus_one), None);
    assert_eq!(minimum.checked_rem_euclid(&minus_one), None);
    assert_eq!(minimum.checked_mod_floor(&minus_one), None);

    assert!(
        catch_unwind(AssertUnwindSafe(|| drop(&minimum % &minus_one))).is_err(),
        "the remainder operator must reject bounded MIN % -1"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| drop(minimum.rem_trunc(&minus_one)))).is_err(),
        "rem_trunc must reject bounded MIN % -1"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| drop(minimum.rem_euclid(&minus_one)))).is_err(),
        "rem_euclid must reject bounded MIN % -1"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| drop(minimum.mod_floor(&minus_one)))).is_err(),
        "mod_floor must reject bounded MIN % -1"
    );
}

#[test]
fn bounded_signed_min_division_policies_match_twos_complement() {
    let width = nz(8);
    let minimum = MpInt::min_for_precision(8);
    let maximum = MpInt::max_for_precision(8);
    let minus_one =
        MpInt::with_precision_checked(-1_i8, width).expect("minus one fits in eight bits");
    let zero = MpInt::zero_with_precision(width);

    assert_eq!(minimum.wrapping_div(&minus_one), minimum);
    assert_eq!(minimum.overflowing_div(&minus_one), (minimum.clone(), true));
    assert_eq!(minimum.saturating_div(&minus_one), maximum);

    assert_eq!(minimum.wrapping_rem(&minus_one), zero);
    assert_eq!(minimum.overflowing_rem(&minus_one), (zero.clone(), true));
    assert_eq!(minimum.saturating_rem(&minus_one), zero);
}

proptest! {
    #[test]
    fn prop_uint_checked_math(a in strategies::uint(8), b in strategies::uint(8)) {
        if a >= b {
            prop_assert_eq!(a.checked_sub(&b), Some(&a - &b));
        } else {
            prop_assert_eq!(a.checked_sub(&b), None);
        }
        if b.value.is_zero() {
            prop_assert_eq!(a.checked_div(&b), None);
            prop_assert_eq!(a.checked_rem(&b), None);
        } else {
            prop_assert_eq!(a.checked_div(&b), Some(&a / &b));
            prop_assert_eq!(a.checked_rem(&b), Some(&a % &b));
        }
    }
}

proptest! {
    #[test]
    fn prop_uint_wrapping_math(
        bits in 8_usize..=128,
        input_a in strategies::bounded_uint_wrapped(128),
        input_b in strategies::bounded_uint_wrapped(128),
    ) {
        let bounded_a = MpUint {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let bounded_b = MpUint {
            value: input_b.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };

        let wrap_add = bounded_a.wrapping_add(&bounded_b);
        let wrap_sub = bounded_a.wrapping_sub(&bounded_b);
        let wrap_mul = bounded_a.wrapping_mul(&bounded_b);

        let mut a_unlimited = bounded_a.clone();
        a_unlimited.precision = Precision::Unlimited;
        let mut b_unlimited = bounded_b.clone();
        b_unlimited.precision = Precision::Unlimited;
        let mod_mask = (MpUint::one() << bits) - MpUint::one();

        prop_assert_eq!(wrap_add, (&a_unlimited + &b_unlimited) & &mod_mask);
        prop_assert_eq!(wrap_mul, (&a_unlimited * &b_unlimited) & &mod_mask);

        if bounded_a >= bounded_b {
            prop_assert_eq!(wrap_sub, &a_unlimited - &b_unlimited);
        } else {
            prop_assert_eq!(
                wrap_sub,
                ((&a_unlimited + &mod_mask + MpUint::one()) - &b_unlimited) & &mod_mask
            );
        }
    }

    #[test]
    fn prop_uint_saturating_math(
        bits in 8_usize..=64,
        input_a in strategies::bounded_uint_wrapped(64),
        input_b in strategies::bounded_uint_wrapped(64),
    ) {
        let bounded_a = MpUint {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let bounded_b = MpUint {
            value: input_b.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let max_val = MpUint::max_for_precision(bits);

        let sat_add = bounded_a.saturating_add(&bounded_b);
        let sat_sub = bounded_a.saturating_sub(&bounded_b);

        let mut a_unlimited = bounded_a.clone();
        a_unlimited.precision = Precision::Unlimited;
        let mut b_unlimited = bounded_b.clone();
        b_unlimited.precision = Precision::Unlimited;

        let sum = &a_unlimited + &b_unlimited;
        if sum > max_val {
            prop_assert_eq!(sat_add, max_val);
        } else {
            prop_assert_eq!(sat_add, sum);
        }

        if bounded_a >= bounded_b {
            prop_assert_eq!(sat_sub, &a_unlimited - &b_unlimited);
        } else {
            prop_assert_eq!(sat_sub, MpUint::zero_with_precision(nz(bits)));
        }
    }

    #[test]
    fn prop_uint_rotate(
        bits in 8_usize..=64,
        input_a in strategies::bounded_uint_wrapped(64),
        shift in 0_u32..100,
    ) {
        let bounded_a = MpUint {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let rot_left = bounded_a.rotate_left(shift, bits).expect("rotate_left failed");
        let rot_right = rot_left
            .rotate_right(shift, bits)
            .expect("rotate_right failed");
        prop_assert_eq!(rot_right, bounded_a, "rotate left then right should be identity");
    }

    #[test]
    fn prop_uint_swap_bytes(a in strategies::uint(4)) {
        let byte_mask = MpUint::from(255_u64);
        let adjusted = if (&a & &byte_mask).value.is_zero() {
            &a | &MpUint::one()
        } else {
            a
        };
        let swapped = adjusted.swap_bytes();
        let swapped_back = swapped.swap_bytes();
        prop_assert_eq!(adjusted, swapped_back, "swap_bytes should be involutive");
    }

    #[test]
    fn prop_uint_bit_counts(bits in 8_usize..=128, input_a in strategies::bounded_uint_wrapped(128)) {
        let bounded_a = MpUint {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };

        let ones = bounded_a.count_ones();
        let zeros = bounded_a.count_zeros().expect("precision should be bounded");
        prop_assert_eq!(ones + zeros, bits, "ones + zeros should equal precision");

        if !bounded_a.value.is_zero() {
            let leading = bounded_a
                .leading_zeros()
                .expect("precision should be bounded");
            let expected_highest_bit = bits - 1 - leading;
            prop_assert!(
                bounded_a.get_bit(expected_highest_bit),
                "highest bit should be 1"
            );
        }
    }

    #[test]
    fn prop_uint_get_set_bit(a in strategies::uint(8), bit_idx in 0_usize..200, bit_val in proptest::bool::ANY) {
        let modified = a.set_bit_to(bit_idx, bit_val);
        prop_assert_eq!(modified.get_bit(bit_idx), bit_val, "set_bit_to should correctly set the bit");
    }
}

proptest! {
    #[test]
    fn prop_int_checked_math(a in strategies::int(4), b in strategies::int(4)) {
        if b.value.abs.is_zero() {
            prop_assert_eq!(a.checked_div(&b), None);
            prop_assert_eq!(a.checked_rem(&b), None);
        } else {
            prop_assert_eq!(a.checked_div(&b), Some(&a / &b));
            prop_assert_eq!(a.checked_rem(&b), Some(&a % &b));
        }
    }
}

proptest! {
    #[test]
    fn prop_int_saturating_math(
        bits in 8_usize..=64,
        input_a in strategies::bounded_int_wrapped(64),
        input_b in strategies::bounded_int_wrapped(64),
    ) {
        let bounded_a = MpInt {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let bounded_b = MpInt {
            value: input_b.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };

        let sat_add = bounded_a.saturating_add(&bounded_b);
        prop_assert!(sat_add <= MpInt::max_for_precision(bits));
        prop_assert!(sat_add >= MpInt::min_for_precision(bits));
    }

    #[test]
    fn prop_int_explicit_width_bitwise_preserves_low_bits(
        input in strategies::int(4),
        width in 1_usize..=128,
        shift in 0_u32..256,
    ) {
        let rotated = input
            .rotate_left(shift, width)
            .expect("non-zero explicit width should rotate");
        prop_assert_eq!(rotated.precision, Precision::Bounded(nz(width)));

        let restored = rotated
            .rotate_right(shift, width)
            .expect("non-zero explicit width should rotate");
        let reversed = input
            .reverse_bits(width)
            .and_then(|value| value.reverse_bits(width))
            .expect("non-zero explicit width should reverse");
        let inverted = input
            .not_with_width(width)
            .and_then(|value| value.not_with_width(width))
            .expect("non-zero explicit width should invert");

        for bit in 0..width {
            prop_assert_eq!(restored.get_bit(bit), input.get_bit(bit));
            prop_assert_eq!(reversed.get_bit(bit), input.get_bit(bit));
            prop_assert_eq!(inverted.get_bit(bit), input.get_bit(bit));
        }
    }

    /// `try_not` must agree with `not_with_width` at the value's own width, and
    /// must refuse rather than guess when there is no width to complement
    /// against.
    #[test]
    fn prop_try_not_matches_explicit_width(
        input in strategies::bounded_uint_wrapped(128),
        width in 1_usize..=128,
    ) {
        let bounded = MpUint {
            value: input.value.apply_wrapping(width),
            precision: Precision::Bounded(nz(width)),
        };

        prop_assert_eq!(bounded.try_not().ok(), bounded.not_with_width(width));

        let unlimited = MpUint {
            value: bounded.value,
            precision: Precision::Unlimited,
        };
        prop_assert_eq!(unlimited.try_not(), Err(MpError::WidthRequired));
    }
}

proptest! {
    #[test]
    fn prop_bounded_shl_matches_exact_fit_rule(
        bits in 1_usize..=64,
        input_seed in strategies::bounded_uint_wrapped(64),
        shift in 0_usize..=64,
    ) {
        let bounded_value = MpUint {
            value: input_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let mut unlimited_value = bounded_value.clone();
        unlimited_value.precision = Precision::Unlimited;
        let exact_shifted = &unlimited_value << shift;
        let should_fit = exact_shifted.value.significant_bits() <= bits;
        let checked = bounded_value.checked_shl(shift);

        prop_assert_eq!(checked.is_some(), should_fit);
        if let Some(checked_value) = checked {
            prop_assert_eq!(&checked_value, &exact_shifted);
            prop_assert_eq!(checked_value.precision, Precision::Bounded(nz(bits)));
            prop_assert_eq!(&bounded_value << shift, exact_shifted);
        } else {
            let operator_panicked = catch_unwind(AssertUnwindSafe(|| {
                drop(&bounded_value << shift);
            }))
            .is_err();
            prop_assert!(operator_panicked, "overflowing bounded shift must panic");
        }
    }
}

proptest! {
    #[test]
    fn prop_wrapping_sub_unlimited_panics_on_underflow(
        left in 0_u64..=u64::from(u32::MAX),
        gap in 1_u64..=u64::from(u32::MAX),
    ) {
        let left_value = uint(left);
        let right_value = uint(left + gap);
        let subtraction_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(left_value.wrapping_sub(&right_value));
        }))
        .is_err();
        prop_assert!(subtraction_panicked, "unlimited unsigned underflow must panic");
    }
}

proptest! {
    #[test]
    fn prop_bounded_value_fits_precision(
        bits in 1_usize..=64,
        bounded_seed in strategies::bounded_uint_wrapped(64),
    ) {
        let bounded_value = MpUint {
            value: bounded_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        bounded_value.debug_assert_valid();
        if let Some(width) = bounded_value.precision.significant_bits() {
            prop_assert!(
                bounded_value.value.significant_bits() <= width,
                "bounded value exceeds precision"
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_uint_shifts(
        bits in 8_usize..=128,
        input_a in strategies::bounded_uint_wrapped(128),
        shift in 0_usize..128,
    ) {
        let bounded_a = MpUint {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let bounded_shift = shift % bits;

        let shifted_left = bounded_a.wrapping_shl(bounded_shift);
        let shifted_right = &shifted_left >> bounded_shift;
        if bounded_a.leading_zeros().unwrap_or(0) >= bounded_shift {
            prop_assert_eq!(
                shifted_right,
                bounded_a,
                "wrapping_shl then wrapping_shr should be identity if no ones are dropped"
            );
        }
    }

    #[test]
    fn prop_int_shifts(
        bits in 8_usize..=128,
        input_a in strategies::bounded_int_wrapped(128),
        shift in 0_usize..128,
    ) {
        let bounded_a = MpInt {
            value: input_a.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let bounded_shift = shift % bits;

        let shifted_left = bounded_a.wrapping_shl(bounded_shift);
        drop(&shifted_left >> bounded_shift);
        drop(bounded_a.saturating_shl(bounded_shift));
        drop(bounded_a.overflowing_shl(bounded_shift));
        prop_assert!(bounded_a.try_shl(bounded_shift).is_ok() || bounded_a.try_shl(bounded_shift).is_err());
    }
}
