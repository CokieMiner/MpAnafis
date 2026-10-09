//! Native oracles for bounded construction, arithmetic policies, and shifts.

extern crate std;

use core::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{MpError, MpInt, MpUint, Precision};

use super::support::{nz, uint};

proptest! {
    #[test]
    fn unsigned_arithmetic_policies_match_native_residues(
        left in any::<u128>(), right in any::<u128>(),
        left_bits in 1_usize..=128, right_bits in 1_usize..=128,
        shift in 0_u32..=130,
    ) {
        let bits = left_bits.max(right_bits);
        let mask = u128::MAX >> (128 - bits);
        let left_mask = u128::MAX >> (128 - left_bits);
        let right_mask = u128::MAX >> (128 - right_bits);
        let a = left & left_mask;
        let b = right & right_mask;
        let width = nz(bits);
        let left_width = nz(left_bits);
        let x = MpUint::with_precision_checked(a, left_width).expect("residue fits");
        let y = MpUint::with_precision_checked(b, nz(right_bits)).expect("residue fits");
        prop_assert_eq!(MpUint::with_precision_wrapping(left, left_width).to_u128(), Some(a));
        prop_assert_eq!(MpUint::with_precision_saturating(left, left_width).to_u128(), Some(left.min(left_mask)));
        prop_assert_eq!(MpUint::with_precision_checked(left, left_width).is_ok(), left <= left_mask);
        let div = a.checked_div(b);
        let rem = a.checked_rem(b);
        let strict_add: fn(&MpUint, &MpUint) -> MpUint = MpUint::strict_add;
        for (checked, tried, wrapping, overflowing, saturating, exact, wrapped, saturated, error, strict) in [
            (x.checked_add(&y), x.try_add(&y), x.wrapping_add(&y), x.overflowing_add(&y), x.saturating_add(&y), a.checked_add(b).filter(|v| *v <= mask), a.wrapping_add(b) & mask, a.saturating_add(b).min(mask), MpError::Overflow, strict_add),
            (x.checked_sub(&y), x.try_sub(&y), x.wrapping_sub(&y), x.overflowing_sub(&y), x.saturating_sub(&y), a.checked_sub(b), a.wrapping_sub(b) & mask, a.saturating_sub(b), MpError::Underflow, MpUint::strict_sub),
            (x.checked_mul(&y), x.try_mul(&y), x.wrapping_mul(&y), x.overflowing_mul(&y), x.saturating_mul(&y), a.checked_mul(b).filter(|v| *v <= mask), a.wrapping_mul(b) & mask, a.saturating_mul(b).min(mask), MpError::Overflow, MpUint::strict_mul),
            (x.checked_div(&y), x.try_div(&y), x.wrapping_div(&y), x.overflowing_div(&y), x.saturating_div(&y), div, div.unwrap_or(0), div.unwrap_or(0), MpError::DivisionByZero, MpUint::strict_div),
            (x.checked_rem(&y), x.try_rem(&y), x.wrapping_rem(&y), x.overflowing_rem(&y), x.saturating_rem(&y), rem, rem.unwrap_or(0), rem.unwrap_or(0), MpError::DivisionByZero, MpUint::strict_rem),
        ] {
            prop_assert_eq!(checked.as_ref().and_then(MpUint::to_u128), exact);
            prop_assert_eq!(tried.as_ref().map_err(|e| *e).map(|v| v.to_u128().expect("residue fits u128")), exact.ok_or(error));
            prop_assert_eq!(wrapping.to_u128(), Some(wrapped));
            prop_assert_eq!(overflowing.0.to_u128(), Some(wrapped));
            prop_assert_eq!(overflowing.1, exact.is_none());
            prop_assert_eq!(saturating.to_u128(), Some(saturated));
            let strict_outcome = catch_unwind(|| strict(&x, &y));
            prop_assert_eq!(strict_outcome.is_err(), exact.is_none());
            prop_assert_eq!(strict_outcome.as_ref().ok().and_then(MpUint::to_u128), exact);
            for result in [Some(wrapping), Some(overflowing.0), Some(saturating), checked, tried.ok(), strict_outcome.ok()].into_iter().flatten() {
                prop_assert_eq!(result.precision(), Precision::Bounded(width));
                prop_assert!(result.significant_bits() <= bits);
            }
        }
        let shift_fits = a == 0 || ((shift as usize) < left_bits && (128 - a.leading_zeros()) as usize + shift as usize <= left_bits);
        let shifted = if shift as usize >= left_bits { 0 } else { a.wrapping_shl(shift) & left_mask };
        prop_assert_eq!(x.wrapping_shl(shift as usize).to_u128(), Some(shifted));
        let (overflowing, overflow) = x.overflowing_shl(shift as usize);
        prop_assert_eq!(overflowing.to_u128(), Some(shifted));
        prop_assert_eq!(overflow, !shift_fits);
        prop_assert_eq!(x.checked_shl(shift as usize).and_then(|v| v.to_u128()), shift_fits.then_some(shifted));
        prop_assert_eq!(x.try_shl(shift as usize).map(|v| v.to_u128().expect("residue fits")), if shift_fits { Ok(shifted) } else { Err(MpError::Overflow) });
        prop_assert_eq!(x.saturating_shl(shift as usize).to_u128(), Some(if shift_fits { shifted } else { left_mask }));
        prop_assert_eq!((&x >> shift as usize).to_u128(), Some(a.checked_shr(shift).unwrap_or(0)));
    }

    #[test]
    fn signed_arithmetic_policies_match_native_twos_complement(
        left in any::<i128>(), right in any::<i128>(),
        left_bits in 1_usize..=128, right_bits in 1_usize..=128, shift in 0_u32..=130,
    ) {
        let bits = left_bits.max(right_bits);
        let padding = u32::try_from(128 - bits).expect("padding below 128");
        let decode = |value: i128| value.wrapping_shl(padding).wrapping_shr(padding);
        let left_padding = u32::try_from(128 - left_bits).expect("padding below 128");
        let right_padding = u32::try_from(128 - right_bits).expect("padding below 128");
        let decode_left = |value: i128| value.wrapping_shl(left_padding).wrapping_shr(left_padding);
        let a = decode_left(left);
        let b = right.wrapping_shl(right_padding).wrapping_shr(right_padding);
        let minimum = i128::MIN >> padding;
        let maximum = !minimum;
        let fits = |value: &i128| (minimum..=maximum).contains(value);
        let width = nz(bits);
        let left_width = nz(left_bits);
        let left_minimum = i128::MIN >> left_padding;
        let left_maximum = !left_minimum;
        let x = MpInt::with_precision_checked(a, left_width).expect("signed residue fits");
        let y = MpInt::with_precision_checked(b, nz(right_bits)).expect("signed residue fits");
        let wrapped_constructor = MpInt::with_precision_wrapping(left, left_width);
        prop_assert_eq!(wrapped_constructor.to_i128(), Some(a));
        prop_assert_eq!(wrapped_constructor.precision(), Precision::Bounded(left_width));
        let saturated_constructor = MpInt::with_precision_saturating(left, left_width);
        prop_assert_eq!(saturated_constructor.to_i128(), Some(left.clamp(left_minimum, left_maximum)));
        prop_assert_eq!(saturated_constructor.precision(), Precision::Bounded(left_width));
        prop_assert_eq!(MpInt::with_precision_checked(left, left_width).is_ok(), (left_minimum..=left_maximum).contains(&left));
        prop_assert_eq!(MpInt::from(left).midpoint(&MpInt::from(right)).to_i128(), Some(left.midpoint(right)));
        let absolute = a.checked_abs().filter(|value| *value <= left_maximum);
        prop_assert_eq!(x.checked_abs().and_then(|value| value.to_i128()), absolute);
        prop_assert_eq!(x.unsigned_abs().to_u128(), Some(a.unsigned_abs()));
        prop_assert_eq!(x.unsigned_abs().precision(), Precision::Bounded(left_width));
        prop_assert_eq!(x.signum().to_i128(), Some(a.signum()));
        prop_assert_eq!(x.signum().precision(), Precision::Bounded(left_width));
        let abs_result = catch_unwind(|| x.abs());
        prop_assert_eq!(abs_result.is_err(), absolute.is_none());
        prop_assert_eq!(abs_result.ok().and_then(|value| value.to_i128()), absolute);
        let mut assigned_abs = x.clone();
        let abs_assignment = catch_unwind(AssertUnwindSafe(|| assigned_abs.abs_assign()));
        prop_assert_eq!(abs_assignment.is_err(), absolute.is_none());
        prop_assert_eq!(assigned_abs.to_i128(), Some(absolute.unwrap_or(a)));
        prop_assert_eq!(assigned_abs.precision(), Precision::Bounded(left_width));
        for negation in [catch_unwind(|| -&x), catch_unwind(|| -x.clone())] {
            prop_assert_eq!(negation.is_err(), absolute.is_none());
            prop_assert_eq!(negation.ok().and_then(|value| value.to_i128()), a.checked_neg().filter(|value| (left_minimum..=left_maximum).contains(value)));
        }
        let positive_difference = if a <= b { Some(0) } else { a.checked_sub(b).filter(fits) };
        let abs_sub = catch_unwind(|| x.abs_sub(&y));
        prop_assert_eq!(abs_sub.is_err(), positive_difference.is_none());
        if let Ok(value) = abs_sub {
            prop_assert_eq!(value.to_i128(), positive_difference);
            prop_assert_eq!(value.precision(), Precision::Bounded(width));
        }
        let division = a.checked_div(b).filter(fits);
        let remainder = division.and_then(|_| a.checked_rem(b));
        let division_error = if b == 0 { MpError::DivisionByZero } else { MpError::Overflow };
        let wrapped_division = if b == 0 { 0 } else { decode(a.wrapping_div(b)) };
        let wrapped_remainder = if b == 0 { 0 } else { a.wrapping_rem(b) };
        let strict_add: fn(&MpInt, &MpInt) -> MpInt = MpInt::strict_add;
        for (checked, tried, wrapping, overflowing, saturating, exact, wrapped, saturated, error, strict) in [
            (x.checked_add(&y), x.try_add(&y), x.wrapping_add(&y), x.overflowing_add(&y), x.saturating_add(&y), a.checked_add(b).filter(fits), decode(a.wrapping_add(b)), a.saturating_add(b).clamp(minimum, maximum), MpError::Overflow, strict_add),
            (x.checked_sub(&y), x.try_sub(&y), x.wrapping_sub(&y), x.overflowing_sub(&y), x.saturating_sub(&y), a.checked_sub(b).filter(fits), decode(a.wrapping_sub(b)), a.saturating_sub(b).clamp(minimum, maximum), MpError::Overflow, MpInt::strict_sub),
            (x.checked_mul(&y), x.try_mul(&y), x.wrapping_mul(&y), x.overflowing_mul(&y), x.saturating_mul(&y), a.checked_mul(b).filter(fits), decode(a.wrapping_mul(b)), a.saturating_mul(b).clamp(minimum, maximum), MpError::Overflow, MpInt::strict_mul),
            (x.checked_div(&y), x.try_div(&y), x.wrapping_div(&y), x.overflowing_div(&y), x.saturating_div(&y), division, wrapped_division, if b == 0 { 0 } else { a.saturating_div(b).clamp(minimum, maximum) }, division_error, MpInt::strict_div),
            (x.checked_rem(&y), x.try_rem(&y), x.wrapping_rem(&y), x.overflowing_rem(&y), x.saturating_rem(&y), remainder, wrapped_remainder, wrapped_remainder, division_error, MpInt::strict_rem),
        ] {
            prop_assert_eq!(checked.as_ref().and_then(MpInt::to_i128), exact);
            prop_assert_eq!(tried.as_ref().map_err(|e| *e).map(|v| v.to_i128().expect("signed residue fits")), exact.ok_or(error));
            prop_assert_eq!(wrapping.to_i128(), Some(wrapped));
            prop_assert_eq!(overflowing.0.to_i128(), Some(wrapped));
            prop_assert_eq!(overflowing.1, exact.is_none());
            prop_assert_eq!(saturating.to_i128(), Some(saturated));
            let strict_outcome = catch_unwind(|| strict(&x, &y));
            prop_assert_eq!(strict_outcome.is_err(), exact.is_none());
            prop_assert_eq!(strict_outcome.as_ref().ok().and_then(MpInt::to_i128), exact);
            for result in [Some(wrapping), Some(overflowing.0), Some(saturating), checked, tried.ok(), strict_outcome.ok()].into_iter().flatten() {
                prop_assert_eq!(result.precision(), Precision::Bounded(width));
                prop_assert!((minimum..=maximum).contains(&result.to_i128().expect("signed residue fits")));
            }
        }
        let exact = (MpInt::zero() + &x) << shift as usize;
        let shift_fits = MpInt::min_for_precision(left_bits) <= exact && exact <= MpInt::max_for_precision(left_bits);
        let wrapped = if shift as usize >= left_bits { 0 } else { decode_left(a.wrapping_shl(shift)) };
        prop_assert_eq!(x.wrapping_shl(shift as usize).to_i128(), Some(wrapped));
        let (shifted, overflow) = x.overflowing_shl(shift as usize);
        prop_assert_eq!(shifted.to_i128(), Some(wrapped));
        prop_assert_eq!(overflow, !shift_fits);
        prop_assert_eq!(x.checked_shl(shift as usize).is_some(), shift_fits);
        prop_assert_eq!(x.try_shl(shift as usize).is_ok(), shift_fits);
        if shift_fits { prop_assert_eq!(&x.try_shl(shift as usize).expect("fits"), &exact); }
        let saturated = if shift_fits { exact } else if a < 0 { MpInt::min_for_precision(left_bits) } else { MpInt::max_for_precision(left_bits) };
        prop_assert_eq!(x.saturating_shl(shift as usize), saturated);
        let expected_right = if shift >= 128 { if a < 0 { -1 } else { 0 } } else { a >> shift };
        prop_assert_eq!((&x >> shift as usize).to_i128(), Some(expected_right));
    }
}

#[test]
fn bounded_subtraction_extends_the_full_residue_width() {
    let residue_bits = 2 * usize::BITS as usize;
    let bits = 3 * usize::BITS as usize;
    let width = nz(bits);
    let unlimited_rhs = (MpUint::one() << residue_bits) - MpUint::one();
    let lhs = MpUint::zero_with_precision(width);
    let rhs = MpUint::with_precision_checked(unlimited_rhs.clone(), width).expect("two limbs fit");
    // For B = 2^usize::BITS, -(B^2 - 1) modulo B^3 equals B^3 - B^2 + 1.
    let expected = (MpUint::one() << bits) - unlimited_rhs;
    let wrapped = lhs.wrapping_sub(&rhs);
    assert_eq!(wrapped, expected);
    assert_eq!(wrapped.precision(), Precision::Bounded(width));
    assert_eq!(lhs.overflowing_sub(&rhs), (wrapped, true));
}

#[test]
fn zero_results_preserve_combined_precision() {
    let expected = Precision::Bounded(nz(16));
    let u = MpUint::with_precision_checked(1_u8, nz(8)).expect("one fits");
    let larger_u = MpUint::with_precision_checked(2_u8, nz(16)).expect("two fits");
    let zero_u = MpUint::zero_with_precision(nz(16));
    for (value, overflow) in [u.overflowing_div(&zero_u), u.overflowing_rem(&zero_u)] {
        assert!(overflow && value.is_zero());
        assert_eq!(value.precision(), expected);
    }
    for value in [
        u.saturating_sub(&larger_u),
        u.saturating_div(&zero_u),
        u.saturating_rem(&zero_u),
    ] {
        assert!(value.is_zero());
        assert_eq!(value.precision(), expected);
    }
    let i = MpInt::with_precision_checked(1_i8, nz(8)).expect("one fits");
    let larger_i = MpInt::with_precision_checked(2_i8, nz(16)).expect("two fits");
    let zero_i = MpInt::zero_with_precision(nz(16));
    for (value, overflow) in [i.overflowing_div(&zero_i), i.overflowing_rem(&zero_i)] {
        assert!(overflow && value.is_zero() && !value.is_negative());
        assert_eq!(value.precision(), expected);
    }
    for value in [
        i.abs_sub(&larger_i),
        i.saturating_div(&zero_i),
        i.saturating_rem(&zero_i),
    ] {
        assert!(value.is_zero() && !value.is_negative());
        assert_eq!(value.precision(), expected);
    }
}

#[test]
fn huge_shift_counts_are_checked_before_allocation() {
    let width = nz(8);
    let u = MpUint::with_precision_checked(1_u8, width).expect("one fits");
    let i = MpInt::with_precision_checked(1_i8, width).expect("one fits");
    let negative = MpInt::with_precision_checked(-1_i8, width).expect("minus one fits");
    let zero_u = MpUint::zero_with_precision(width);
    let zero_i = MpInt::zero_with_precision(width);
    assert_eq!(zero_u.checked_shl(usize::MAX), Some(zero_u.clone()));
    assert_eq!(zero_u.try_shl(usize::MAX), Ok(zero_u.clone()));
    assert_eq!(zero_u.overflowing_shl(usize::MAX), (zero_u.clone(), false));
    assert_eq!(zero_i.checked_shl(usize::MAX), Some(zero_i.clone()));
    assert_eq!(zero_i.try_shl(usize::MAX), Ok(zero_i.clone()));
    assert_eq!(zero_i.overflowing_shl(usize::MAX), (zero_i.clone(), false));
    assert_eq!(u.checked_shl(usize::MAX), None);
    assert_eq!(i.checked_shl(usize::MAX), None);
    assert_eq!(u.try_shl(usize::MAX), Err(MpError::Overflow));
    assert_eq!(i.try_shl(usize::MAX), Err(MpError::Overflow));
    assert_eq!(u.wrapping_shl(usize::MAX), zero_u);
    assert_eq!(i.wrapping_shl(usize::MAX), zero_i);
    assert_eq!(
        u.overflowing_shl(usize::MAX),
        (MpUint::zero_with_precision(width), true)
    );
    assert_eq!(
        i.overflowing_shl(usize::MAX),
        (MpInt::zero_with_precision(width), true)
    );
    assert_eq!(u.saturating_shl(usize::MAX), MpUint::max_for_precision(8));
    assert_eq!(i.saturating_shl(usize::MAX), MpInt::max_for_precision(8));
    assert_eq!(
        negative.saturating_shl(usize::MAX),
        MpInt::min_for_precision(8)
    );
    assert_eq!(
        negative.wrapping_shl(usize::MAX),
        MpInt::zero_with_precision(width)
    );
    assert_eq!(negative.checked_shl(7), Some(MpInt::min_for_precision(8)));
    assert_eq!(i.checked_shl(6).and_then(|v| v.to_i128()), Some(64));
    for bits in [1, 2, 63, 64, 65, 255, 256, 257, 4096] {
        assert_bounded_shift_validation(bits);
    }
}

#[test]
fn signed_minimum_division_policies_share_the_overflow_contract() {
    for bits in [1, 2, 8, 64, 65, 128, 256] {
        let minimum = MpInt::min_for_precision(bits);
        let minus_one = MpInt::with_precision_checked(-1_i8, nz(bits)).expect("minus one fits");
        let zero = MpInt::zero_with_precision(nz(bits));
        assert_eq!(minimum.checked_rem(&minus_one), None);
        assert_eq!(minimum.try_rem(&minus_one), Err(MpError::Overflow));
        assert_eq!(minimum.div_rem(&minus_one), None);
        assert_eq!(minimum.checked_rem_trunc(&minus_one), None);
        assert_eq!(minimum.checked_rem_euclid(&minus_one), None);
        assert_eq!(minimum.checked_mod_floor(&minus_one), None);
        assert!(catch_unwind(|| &minimum % &minus_one).is_err());
        assert!(catch_unwind(|| minimum.rem_trunc(&minus_one)).is_err());
        assert!(catch_unwind(|| minimum.rem_euclid(&minus_one)).is_err());
        assert!(catch_unwind(|| minimum.mod_floor(&minus_one)).is_err());
        assert_eq!(minimum.wrapping_div(&minus_one), minimum);
        assert_eq!(minimum.overflowing_div(&minus_one), (minimum.clone(), true));
        assert_eq!(
            minimum.saturating_div(&minus_one),
            MpInt::max_for_precision(bits)
        );
        assert_eq!(minimum.wrapping_rem(&minus_one), zero);
        assert_eq!(minimum.overflowing_rem(&minus_one), (zero.clone(), true));
        assert_eq!(minimum.saturating_rem(&minus_one), zero);
    }
}

proptest! {
    #[test]
    fn unlimited_unsigned_wrapping_subtraction_rejects_underflow(
        left in 0_u64..=u64::from(u32::MAX), gap in 1_u64..=u64::from(u32::MAX),
    ) {
        let a = uint(left);
        let b = uint(left + gap);
        prop_assert!(catch_unwind(AssertUnwindSafe(|| a.wrapping_sub(&b))).is_err());
    }
}

fn assert_bounded_shift_validation(bits: usize) {
    let width = nz(bits);
    let unsigned = MpUint::with_precision_checked(1_u8, width).expect("one fits unsigned width");
    let signed = MpInt::with_precision_checked(-1_i8, width).expect("minus one fits signed width");
    for shift in [bits, usize::MAX] {
        let operations: [&dyn Fn(); 6] = [
            &|| drop(&unsigned << shift),
            &|| drop(unsigned.clone() << shift),
            &|| drop(unsigned.mul_2exp(shift)),
            &|| drop(&signed << shift),
            &|| drop(signed.clone() << shift),
            &|| drop(signed.mul_2exp(shift)),
        ];
        for operation in operations {
            let failure =
                catch_unwind(AssertUnwindSafe(operation)).expect_err("shift exceeds bounded width");
            let message = failure
                .downcast_ref::<alloc::string::String>()
                .map(alloc::string::String::as_str)
                .or_else(|| failure.downcast_ref::<&str>().copied())
                .expect("string panic");
            assert!(
                message.contains("Bounded"),
                "bounded validation precedes allocation"
            );
        }
    }
    let unsigned_zero = MpUint::zero_with_precision(width);
    let signed_zero = MpInt::zero_with_precision(width);
    for result in [
        &unsigned_zero << usize::MAX,
        unsigned_zero.clone() << usize::MAX,
        unsigned_zero.mul_2exp(usize::MAX),
    ] {
        assert!(result.is_zero());
        assert_eq!(result.precision(), Precision::Bounded(width));
    }
    for result in [
        &signed_zero << usize::MAX,
        signed_zero.clone() << usize::MAX,
        signed_zero.mul_2exp(usize::MAX),
    ] {
        assert!(result.is_zero() && !result.is_negative());
        assert_eq!(result.precision(), Precision::Bounded(width));
    }
}
