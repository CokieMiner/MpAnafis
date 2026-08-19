//! Broad no-panic coverage for the public integer API surface.

use super::*;

proptest! {
    #[test]
    fn prop_new_public_apis_basic(a in strategies::uint(16), b in strategies::uint(16), c in strategies::int(16), d in strategies::int(16)) {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<MpUint>();
        assert_sync::<MpUint>();
        assert_send::<MpInt>();
        assert_sync::<MpInt>();

        let mut u1 = a; let mut u2 = b;
        let u1_orig = u1.clone(); let u2_orig = u2.clone();
        u1.swap(&mut u2);
        prop_assert_eq!(u1, u2_orig); prop_assert_eq!(u2, u1_orig);

        let mut i1 = c; let mut i2 = d;
        let i1_orig = i1.clone(); let i2_orig = i2.clone();
        i1.swap(&mut i2);
        prop_assert_eq!(i1, i2_orig); prop_assert_eq!(i2, i1_orig);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    #[allow(
        clippy::let_underscore_must_use,
        let_underscore_drop,
        reason = "API exhaustion property: exercises all public methods; results intentionally discarded"
    )]
    fn prop_api_exhaustion_uint(
        bits_a in 8_usize..=128,
        bits_b in 8_usize..=128,
        input_a in strategies::uint(2),
        input_b in strategies::uint(2),
        make_bounded in any::<bool>(),
        shift in 0_usize..100_usize,
    ) {
        let mut left_value = input_a;
        let mut right_value = input_b;

        if make_bounded {
            left_value.value = left_value.value.apply_wrapping(bits_a);
            right_value.value = right_value.value.apply_wrapping(bits_b);
            left_value.precision = Precision::Bounded(nz(bits_a));
            right_value.precision = Precision::Bounded(nz(bits_b));
        }

        drop(left_value.checked_add(&right_value));
        drop(left_value.wrapping_add(&right_value));
        drop(left_value.saturating_add(&right_value));
        drop(left_value.overflowing_add(&right_value));

        drop(left_value.checked_sub(&right_value));
        if make_bounded {
            drop(left_value.wrapping_sub(&right_value));
        }
        drop(left_value.saturating_sub(&right_value));
        drop(left_value.overflowing_sub(&right_value));

        drop(left_value.checked_mul(&right_value));
        drop(left_value.wrapping_mul(&right_value));
        drop(left_value.saturating_mul(&right_value));
        drop(left_value.overflowing_mul(&right_value));

        if !right_value.is_zero() {
            drop(left_value.checked_div(&right_value));
            drop(left_value.overflowing_div(&right_value));
            drop(left_value.checked_rem(&right_value));
            drop(left_value.overflowing_rem(&right_value));
        }

        drop(left_value.checked_shl(shift));
        if make_bounded {
            drop(left_value.wrapping_shl(shift));
        }
        drop(left_value.saturating_shl(shift));
        drop(left_value.overflowing_shl(shift));

        drop(&left_value >> shift);

        let _ = left_value.is_even();
        let _ = left_value.is_odd();
        let _ = left_value.is_power_of_two();
        let _ = left_value.is_one();
        let _ = left_value.is_zero();

        let _ = left_value.to_u64();
        let _ = left_value.to_u128();
        let _ = left_value.to_usize();
        let _ = left_value.to_i64();
        let _ = left_value.to_i128();
        let _ = left_value.to_isize();
        let _ = left_value.to_f32();
        let _ = left_value.to_f64();

    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    #[allow(
        clippy::let_underscore_must_use,
        let_underscore_drop,
        reason = "API exhaustion property: exercises all public methods; results intentionally discarded"
    )]
    fn prop_api_exhaustion_int(
        bits_a in 8_usize..=128,
        bits_b in 8_usize..=128,
        input_a in strategies::int(2),
        input_b in strategies::int(2),
        make_bounded in any::<bool>(),
        shift in 0_usize..100_usize,
    ) {
        let mut left_value = input_a;
        let mut right_value = input_b;

        if make_bounded {
            left_value.value = left_value.value.apply_wrapping(bits_a);
            right_value.value = right_value.value.apply_wrapping(bits_b);
            left_value.precision = Precision::Bounded(nz(bits_a));
            right_value.precision = Precision::Bounded(nz(bits_b));
        }

        drop(left_value.checked_add(&right_value));
        drop(left_value.wrapping_add(&right_value));
        drop(left_value.saturating_add(&right_value));
        drop(left_value.overflowing_add(&right_value));

        drop(left_value.checked_sub(&right_value));
        if make_bounded {
            drop(left_value.wrapping_sub(&right_value));
        }
        drop(left_value.saturating_sub(&right_value));
        drop(left_value.overflowing_sub(&right_value));

        drop(left_value.checked_mul(&right_value));
        drop(left_value.wrapping_mul(&right_value));
        drop(left_value.saturating_mul(&right_value));
        drop(left_value.overflowing_mul(&right_value));

        if !right_value.is_zero() {
            drop(left_value.checked_div(&right_value));
            drop(left_value.overflowing_div(&right_value));
            drop(left_value.checked_rem(&right_value));
            drop(left_value.overflowing_rem(&right_value));
        }

        if make_bounded {
            drop(left_value.wrapping_shl(shift));
        }
        drop(left_value.saturating_shl(shift));
        drop(left_value.overflowing_shl(shift));

        drop(&left_value >> shift);

        let _ = left_value.is_even();
        let _ = left_value.is_odd();
        let _ = left_value.is_power_of_two();
        let _ = left_value.is_one();
        let _ = left_value.is_zero();

        let _ = left_value.to_u64();
        let _ = left_value.to_u128();
        let _ = left_value.to_usize();
        let _ = left_value.to_i64();
        let _ = left_value.to_i128();
        let _ = left_value.to_isize();
        let _ = left_value.to_f32();
        let _ = left_value.to_f64();

        let checked_abs = left_value.checked_abs();
        if checked_abs.is_some() {
            drop(left_value.abs());
        }
        drop(left_value.signum());
        if checked_abs.is_some() {
            left_value.abs_assign();
        }
    }
}
