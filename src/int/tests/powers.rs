//! Exponentiation and integer-root properties.

use core::panic::AssertUnwindSafe;

use super::{std::panic::catch_unwind, *};

proptest! {
    #[test]
    fn prop_pow_identity(a in strategies::uint(8), exponent in 0_u32..=128) {
        prop_assert_eq!(a.pow(0_u32), MpUint::one(), "a^0 != 1");
        prop_assert!(a.pow(1_u32) == a, "a^1 != a");
        prop_assert_eq!(MpUint::zero().pow(0_u32), MpUint::one(), "0^0 != 1");
        if exponent == 0 {
            prop_assert_eq!(MpUint::zero().pow(exponent), MpUint::one());
        } else {
            prop_assert_eq!(MpUint::zero().pow(exponent), MpUint::zero());
        }
        prop_assert_eq!(MpUint::one().pow(exponent), MpUint::one());
    }
}

proptest! {
    #[test]
    fn prop_pow_mul(a in strategies::uint(4), e1 in 0_u32..=5, e2 in 0_u32..=5) {
        prop_assert_eq!(a.pow(e1 + e2), &a.pow(e1) * &a.pow(e2), "a^(e1+e2) != a^e1 * a^e2");
    }
}

proptest! {
    #[test]
    fn prop_isqrt_square(a in strategies::uint(4)) {
        let sq = &a * &a;
        let root = sq.isqrt().expect("isqrt should succeed");
        prop_assert_eq!(root, a, "isqrt(a^2) != a");
    }
}

proptest! {
    #[test]
    fn prop_isqrt_lower_bound(a in strategies::uint(4)) {
        let root = a.isqrt().expect("isqrt should succeed");
        let root_sq = &root * &root;
        let next_sq = &(&root + &MpUint::one()) * &(&root + &MpUint::one());
        prop_assert!(root_sq <= a, "isqrt^2 > a");
        prop_assert!(a < next_sq, "isqrt too small");
    }
}

proptest! {
    #[test]
    fn prop_new_public_apis_pow_and_isqrt(
        signed_value in strategies::int(1),
        unsigned_value in strategies::uint(1),
        exponent in 0_u32..=10_u32,
    ) {
        let signed_power = signed_value.pow(exponent);
        prop_assert_eq!(signed_value.checked_pow(exponent), Some(signed_power.clone()));
        prop_assert_eq!(signed_value.try_pow(exponent), Ok(signed_power.clone()));

        if exponent == 0 {
            prop_assert_eq!(signed_power, MpInt::from(1_u8));
        } else if exponent == 1 {
            prop_assert_eq!(&signed_power, &signed_value);
        } else if exponent == 2 {
            prop_assert_eq!(signed_power, &signed_value * &signed_value);
        }

        if signed_value.is_negative() {
            prop_assert_eq!(signed_value.checked_isqrt(), None);
        } else {
            let squared_value = &signed_value * &signed_value;
            prop_assert_eq!(squared_value.checked_isqrt(), Some(signed_value.abs()));
        }

        let unsigned_power = unsigned_value.pow(exponent);
        prop_assert_eq!(unsigned_value.checked_pow(exponent), Some(unsigned_power.clone()));
        prop_assert_eq!(unsigned_value.try_pow(exponent), Ok(unsigned_power));
    }
}

proptest! {
    #[test]
    fn prop_bounded_precision_pow_matches_exact_fit(
        bits in 2_usize..=16,
        signed_seed in strategies::bounded_int_wrapped(16),
        unsigned_seed in strategies::bounded_uint_wrapped(16),
        exponent in 0_u32..=8,
    ) {
        let bounded_signed = MpInt {
            value: signed_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let mut unlimited_signed = bounded_signed.clone();
        unlimited_signed.precision = Precision::Unlimited;
        let exact_signed = unlimited_signed.pow(exponent);
        let signed_should_fit =
            exact_signed.value.required_signed_bits_for_bounded_storage() <= bits;
        let signed_checked = bounded_signed.checked_pow(exponent);
        let signed_try = bounded_signed.try_pow(exponent);
        let signed_pow_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(bounded_signed.pow(exponent));
        }))
        .is_err();
        prop_assert_eq!(signed_checked.is_some(), signed_should_fit);
        prop_assert_eq!(signed_pow_panicked, !signed_should_fit);
        if let Some(checked_value) = signed_checked {
            prop_assert_eq!(&checked_value, &exact_signed);
            prop_assert_eq!(checked_value.precision, Precision::Bounded(nz(bits)));
            let tried_value = signed_try.expect("fitting signed power");
            prop_assert_eq!(&tried_value, &exact_signed);
            prop_assert_eq!(tried_value.precision, Precision::Bounded(nz(bits)));
        } else {
            prop_assert_eq!(signed_try, Err(MpError::Overflow));
        }

        let signed_square_checked = bounded_signed.checked_mul(&bounded_signed);
        let signed_square_try = bounded_signed.try_mul(&bounded_signed);
        let signed_square_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(bounded_signed.square());
        }))
        .is_err();
        prop_assert_eq!(signed_square_panicked, signed_square_checked.is_none());
        prop_assert_eq!(
            signed_square_panicked,
            signed_square_try == Err(MpError::Overflow)
        );

        let bounded_unsigned = MpUint {
            value: unsigned_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let mut unlimited_unsigned = bounded_unsigned.clone();
        unlimited_unsigned.precision = Precision::Unlimited;
        let exact_unsigned = unlimited_unsigned.pow(exponent);
        let unsigned_should_fit = exact_unsigned.value.significant_bits() <= bits;
        let unsigned_checked = bounded_unsigned.checked_pow(exponent);
        let unsigned_try = bounded_unsigned.try_pow(exponent);
        let unsigned_pow_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(bounded_unsigned.pow(exponent));
        }))
        .is_err();
        prop_assert_eq!(unsigned_checked.is_some(), unsigned_should_fit);
        prop_assert_eq!(unsigned_pow_panicked, !unsigned_should_fit);
        if let Some(checked_value) = unsigned_checked {
            prop_assert_eq!(&checked_value, &exact_unsigned);
            prop_assert_eq!(checked_value.precision, Precision::Bounded(nz(bits)));
            let tried_value = unsigned_try.expect("fitting unsigned power");
            prop_assert_eq!(&tried_value, &exact_unsigned);
            prop_assert_eq!(tried_value.precision, Precision::Bounded(nz(bits)));
        } else {
            prop_assert_eq!(unsigned_try, Err(MpError::Overflow));
        }

        let unsigned_square_checked = bounded_unsigned.checked_mul(&bounded_unsigned);
        let unsigned_square_try = bounded_unsigned.try_mul(&bounded_unsigned);
        let unsigned_square_panicked = catch_unwind(AssertUnwindSafe(|| {
            drop(bounded_unsigned.square());
        }))
        .is_err();
        prop_assert_eq!(unsigned_square_panicked, unsigned_square_checked.is_none());
        prop_assert_eq!(
            unsigned_square_panicked,
            unsigned_square_try == Err(MpError::Overflow)
        );
    }

    #[test]
    fn prop_checked_next_power_of_two_preserves_bounded_precision(
        bits in 1_usize..=128,
        input_seed in strategies::bounded_uint_wrapped(128),
    ) {
        let bounded_zero = MpUint {
            value: InternalMpUint::zero(),
            precision: Precision::Bounded(nz(bits)),
        };
        let zero_next = bounded_zero
            .checked_next_power_of_two()
            .expect("one fits every valid unsigned precision");
        prop_assert_eq!(&zero_next, &MpUint::one());
        prop_assert_eq!(zero_next.precision, Precision::Bounded(nz(bits)));

        let bounded_value = MpUint {
            value: input_seed.value.apply_wrapping(bits),
            precision: Precision::Bounded(nz(bits)),
        };
        let mut exact_next = if bounded_value.is_zero() {
            MpUint::one()
        } else if bounded_value.is_power_of_two() {
            bounded_value.clone()
        } else {
            MpUint::one() << bounded_value.significant_bits()
        };
        exact_next.precision = Precision::Unlimited;
        let should_fit = exact_next.value.significant_bits() <= bits;
        let checked_next = bounded_value.checked_next_power_of_two();

        prop_assert_eq!(checked_next.is_some(), should_fit);
        if let Some(next_value) = checked_next {
            prop_assert_eq!(&next_value, &exact_next);
            prop_assert_eq!(next_value.precision, Precision::Bounded(nz(bits)));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_factorial_honors_requested_bounded_precision(
        n in 0_u32..=50,
        bits in 1_usize..=256,
    ) {
        let unsigned_exact = MpUint::factorial(n, Precision::Unlimited);
        let unsigned_should_fit =
            unsigned_exact.value.required_unsigned_bits_for_bounded_storage() <= bits;
        let unsigned_bounded = catch_unwind(AssertUnwindSafe(|| {
            MpUint::factorial(n, Precision::Bounded(nz(bits)))
        }));
        prop_assert_eq!(unsigned_bounded.is_ok(), unsigned_should_fit);
        if let Ok(value) = unsigned_bounded {
            prop_assert_eq!(&value, &unsigned_exact);
            prop_assert_eq!(value.precision, Precision::Bounded(nz(bits)));
        }

        let signed_exact = MpInt::factorial(n, Precision::Unlimited);
        let signed_should_fit =
            signed_exact.value.required_signed_bits_for_bounded_storage() <= bits;
        let signed_bounded = catch_unwind(AssertUnwindSafe(|| {
            MpInt::factorial(n, Precision::Bounded(nz(bits)))
        }));
        prop_assert_eq!(signed_bounded.is_ok(), signed_should_fit);
        if let Ok(value) = signed_bounded {
            prop_assert_eq!(&value, &signed_exact);
            prop_assert_eq!(value.precision, Precision::Bounded(nz(bits)));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn stress_large_pow_and_fermat(
        large_base in 2_u32..=10,
        exponent in 128_u32..=512,
        modular_base in 1_u32..=10_000,
        prime in prop_oneof![Just(3_u32), Just(5), Just(17), Just(97)],
    ) {
        let base = MpUint::from(large_base);
        let large_power = base.pow(exponent);
        let previous_power = base.pow(exponent - 1);
        prop_assert_eq!(large_power, previous_power * &base);

        let prime_value = MpUint::from(prime);
        let fermat = MpUint::from(modular_base)
            .pow_mod(&MpUint::from(prime - 1), &prime_value)
            .expect("prime modulus is non-zero");
        let expected = if modular_base.is_multiple_of(prime) {
            MpUint::zero()
        } else {
            MpUint::one()
        };
        prop_assert_eq!(fermat, expected, "Fermat residue mismatch");
    }
}
