//! Expected arithmetic panics and receiver rollback outside libFuzzer's abort hook.

use std::panic::{AssertUnwindSafe, catch_unwind};

use mp_anafis::{BoundedPrecision, MpInt, MpUint, Precision};

#[test]
fn assignment_failures_preserve_value_and_destination_precision() {
    macro_rules! rollback {
        ($ty:ty, $initial:expr, $operation:expr) => {{
            let mut value =
                <$ty>::with_precision_checked($initial, BoundedPrecision::new(2).unwrap()).unwrap();
            let before = value.clone();
            assert!(catch_unwind(AssertUnwindSafe(|| ($operation)(&mut value))).is_err());
            assert_eq!(value, before);
            assert_eq!(value.precision(), before.precision());
        }};
    }
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value +=
        MpUint::from(4_u8));
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value -=
        MpUint::from(4_u8));
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value *=
        MpUint::from(4_u8));
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value /= MpUint::zero());
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value %= MpUint::zero());
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value <<= 2_u8);
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value |=
        MpUint::from(4_u8));
    rollback!(MpUint, 3_u8, |value: &mut MpUint| *value ^=
        MpUint::from(4_u8));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value += MpInt::from(4_i8));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value -= MpInt::from(4_i8));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value *= MpInt::from(4_i8));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value /= MpInt::zero());
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value %= MpInt::zero());
    rollback!(MpInt, -2_i8, |value: &mut MpInt| *value %=
        MpInt::minus_one());
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value <<= 2_u8);
    rollback!(MpInt, -2_i8, |value: &mut MpInt| *value &=
        MpInt::from(-4_i8));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value |= MpInt::from(4_i8));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| *value ^= MpInt::from(4_i8));
    rollback!(MpInt, -2_i8, MpInt::abs_assign);
    rollback!(MpUint, 3_u8, |value: &mut MpUint| value
        .assign_add(&MpUint::from(3_u8), &MpUint::one()));
    rollback!(MpUint, 3_u8, |value: &mut MpUint| value
        .assign_mul(&MpUint::from(2_u8), &MpUint::from(2_u8)));
    rollback!(MpUint, 3_u8, |value: &mut MpUint| value
        .assign_square(&MpUint::from(2_u8)));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| value
        .assign_add(&MpInt::one(), &MpInt::one()));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| value
        .assign_sub(&MpInt::from(-2_i8), &MpInt::one()));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| value
        .assign_mul(&MpInt::from(2_i8), &MpInt::one()));
    rollback!(MpInt, 1_i8, |value: &mut MpInt| value
        .assign_square(&MpInt::from(2_i8)));
}

#[test]
fn panicking_boundaries_reject_invalid_counts_widths_and_domains() {
    let unsigned = MpUint::one();
    let signed = MpInt::one();
    macro_rules! negative_counts {
        ($($ty:ty),+ $(,)?) => {$(
            let count: $ty = -1;
            assert!(catch_unwind(|| &unsigned << count).is_err()); assert!(catch_unwind(|| &unsigned >> count).is_err());
            assert!(catch_unwind(|| &signed << count).is_err()); assert!(catch_unwind(|| &signed >> count).is_err());
        )+};
    }
    negative_counts!(i8, i16, i32, i64, i128, isize);
    assert!(catch_unwind(|| &unsigned << u128::MAX).is_err());
    assert!(catch_unwind(|| &signed >> u128::MAX).is_err());
    assert!(catch_unwind(|| !unsigned.clone()).is_err());
    assert!(catch_unwind(|| !&unsigned).is_err());
    assert!(catch_unwind(|| MpUint::zero().wrapping_sub(&unsigned)).is_err());
    for radix in [0, 1, 37, u32::MAX] {
        assert!(catch_unwind(|| unsigned.to_string_radix(radix)).is_err());
        assert!(catch_unwind(|| signed.to_string_radix(radix)).is_err());
    }
    for bits in [0, usize::MAX] {
        assert!(catch_unwind(|| MpUint::max_for_precision(bits)).is_err());
        assert!(catch_unwind(|| MpUint::min_for_precision(bits)).is_err());
        assert!(catch_unwind(|| MpInt::max_for_precision(bits)).is_err());
        assert!(catch_unwind(|| MpInt::min_for_precision(bits)).is_err());
    }
    let width = BoundedPrecision::new(1).unwrap();
    let minimum = MpInt::with_precision_checked(-1_i8, width).unwrap();
    assert!(catch_unwind(|| -&minimum).is_err());
    assert!(catch_unwind(|| minimum.abs()).is_err());
    assert!(catch_unwind(|| minimum.gcd(&MpInt::zero_with_precision(width))).is_err());
    assert!(catch_unwind(|| minimum.pow(0)).is_err());
    assert!(catch_unwind(|| MpInt::factorial(0, Precision::Bounded(width))).is_err());
    let maximum = MpUint::with_precision_checked(1_u8, width).unwrap();
    assert!(catch_unwind(|| maximum.strict_add(&maximum)).is_err());
    assert!(catch_unwind(|| maximum.strict_sub(&MpUint::from(2_u8))).is_err());
    assert!(catch_unwind(|| maximum.strict_div(&MpUint::zero())).is_err());
    assert!(catch_unwind(|| maximum.strict_rem(&MpUint::zero())).is_err());
}
