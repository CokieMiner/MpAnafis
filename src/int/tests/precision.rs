//! Precision propagation, ambient context, and representation invariant properties.

use super::*;

proptest! {
    #[test]
    fn bounded_widths_have_one_canonical_domain(
        bits in prop_oneof![Just(0), Just(usize::MAX), any::<usize>()],
    ) {
        let width = BoundedPrecision::new(bits);
        let should_be_bounded = (1..usize::MAX).contains(&bits);

        prop_assert_eq!(width.is_some(), should_be_bounded);
        prop_assert_eq!(Precision::new_bounded(bits).is_some(), should_be_bounded);
        prop_assert_eq!(AmbientPrecision::new_bounded(bits).is_some(), should_be_bounded);

        if let Some(valid_width) = width {
            prop_assert_eq!(valid_width.get(), bits);
            prop_assert_eq!(
                Precision::new_bounded(bits),
                Some(Precision::Bounded(valid_width))
            );
            prop_assert_eq!(
                AmbientPrecision::new_bounded(bits),
                Some(AmbientPrecision::Bounded(valid_width))
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_eq_hash_consistency(a in strategies::uint(16)) {
        let b = ArbiUint {
            value: a.value.clone(),
            precision: Precision::Unlimited,
        };
        prop_assert_eq!(&a, &b, "same value, different precision should be Eq");
        #[cfg(feature = "std")]
        prop_assert_eq!(hash_u64(&a), hash_u64(&b), "same value, different precision should hash same");
    }
}

proptest! {
    #[test]
    fn prop_ord_implies_eq_consistency(x in strategies::uint(12), y in strategies::uint(12)) {
        if x.cmp(&y) == Ordering::Equal {
            prop_assert_eq!(&x, &y, "Ord::Equal but not Eq");
            #[cfg(feature = "std")]
            prop_assert_eq!(hash_u64(&x), hash_u64(&y), "Ord::Equal but different hash");
        }
    }
}

proptest! {
    #[test]
    fn prop_assign_preserves_lhs_precision(
        bits in 8_usize..=64,
        left_seed in strategies::bounded_uint_wrapped(64),
        right_seed in strategies::bounded_uint_wrapped(64),
    ) {
        let max_val = (ArbiUint::one() << bits) - ArbiUint::one();
        let left_value = left_seed.value.apply_wrapping(bits);
        let right_value = right_seed.value.apply_wrapping(bits);
        let mut left_operand = ArbiUint {
            value: left_value.clone(),
            precision: Precision::Bounded(nz(bits)),
        };
        let right_operand = ArbiUint {
            value: right_value.clone(),
            precision: Precision::Unlimited,
        };
        let expected_precision = left_operand.precision;
        let left_unlimited = ArbiUint {
            value: left_value,
            precision: Precision::Unlimited,
        };
        let right_unlimited = ArbiUint {
            value: right_value,
            precision: Precision::Unlimited,
        };

        if &left_unlimited + &right_unlimited <= max_val {
            left_operand += &right_operand;
            prop_assert_eq!(
                left_operand.precision,
                expected_precision,
                "assignment should preserve the left-hand side precision"
            );
        }
    }

    #[test]
    fn prop_bounded_result_max_width(
        width_a in 8_usize..=64,
        width_b in 8_usize..=64,
        left_seed in strategies::bounded_uint_wrapped(64),
        right_seed in strategies::bounded_uint_wrapped(64),
    ) {
        let left_operand = ArbiUint {
            value: left_seed.value.apply_wrapping(width_a),
            precision: Precision::Bounded(nz(width_a)),
        };
        let right_operand = ArbiUint {
            value: right_seed.value.apply_wrapping(width_b),
            precision: Precision::Bounded(nz(width_b)),
        };
        let expected_max = width_a.max(width_b);
        let sum = left_operand.wrapping_add(&right_operand);
        if let Some(result_width) = sum.precision.significant_bits() {
            prop_assert_eq!(
                result_width,
                expected_max,
                "bounded+bounded result width should be max(wa, wb)"
            );
        }
    }
}

proptest! {
    #[test]
    fn prop_checked_no_overflow_eq_plain(a in strategies::uint(8), b in strategies::uint(8)) {
        if let Some(sum) = a.checked_add(&b) {
            prop_assert_eq!(sum, &a + &b, "checked_add != plain add when no overflow");
        }
        if let Some(prod) = a.checked_mul(&b) {
            prop_assert_eq!(prod, &a * &b, "checked_mul != plain mul when no overflow");
        }
        if a >= b && let Some(diff) = a.checked_sub(&b) {
            prop_assert_eq!(diff, &a - &b, "checked_sub != plain sub when no underflow");
        }
    }
}

proptest! {
    #[test]
    fn ambient_unsigned_no_context(
        value in any::<u64>(),
        small_value in any::<u8>(),
        medium_value in any::<u16>(),
    ) {
        prop_assert_eq!(ArbiUint::from(value).precision, Precision::Unlimited);
        prop_assert_eq!(ArbiUint::from(small_value).precision, Precision::Unlimited);
        prop_assert_eq!(ArbiUint::from(medium_value).precision, Precision::Unlimited);
        prop_assert_eq!(ArbiUint::default().precision, Precision::Unlimited);
        let _ = &PrecisionContext;
    }
}

#[cfg(feature = "std")]
proptest! {
    #[test]
    fn ambient_bounded_construction_uses_maximum_required_width(
        bits in 1_usize..=128,
        unsigned_value in any::<u128>(),
        signed_value in any::<i128>(),
    ) {
        PrecisionContext::with_bounded(bits, || {
            let unsigned = ArbiUint::from(unsigned_value);
            let expected_unsigned_bits =
                bits.max(unsigned.value.required_unsigned_bits_for_bounded_storage());
            prop_assert_eq!(
                unsigned.precision,
                Precision::Bounded(nz(expected_unsigned_bits))
            );

            let signed = ArbiInt::from(signed_value);
            let expected_signed_bits =
                bits.max(signed.value.required_signed_bits_for_bounded_storage());
            prop_assert_eq!(
                signed.precision,
                Precision::Bounded(nz(expected_signed_bits))
            );
            prop_assert_eq!(ArbiUint::default().precision, Precision::Unlimited);
            Ok(())
        })?;
    }

    #[test]
    fn ambient_unsigned_from_str_respects_bound(
        bits in 1_usize..=128,
        input in any::<u128>(),
    ) {
        let encoded = input.to_string();
        let required_bits = InternalArbiUint::from_u128(input)
            .required_unsigned_bits_for_bounded_storage();
        PrecisionContext::with_bounded(bits, || {
            let parsed = encoded.parse::<ArbiUint>();
            prop_assert_eq!(parsed.is_ok(), required_bits <= bits);
            if let Ok(value) = parsed {
                prop_assert_eq!(value.precision, Precision::Bounded(nz(bits)));
                prop_assert_eq!(value.to_u128(), Some(input));
            }
            Ok(())
        })?;
    }

    #[test]
    fn ambient_assign_preserves_lhs_precision(
        bits in 8_usize..=64,
        lhs in 0_u8..=100,
        rhs in 0_u8..=100,
    ) {
        let mut left_value = ArbiUint::with_precision_checked(lhs, nz(bits)).expect("fits");
        let right_value = ArbiUint::from(rhs);
        left_value += &right_value;
        prop_assert_eq!(left_value.precision, Precision::Bounded(nz(bits)));
        prop_assert_eq!(
            left_value.to_u64(),
            Some(u64::from(lhs).wrapping_add(u64::from(rhs)))
        );
    }

    #[test]
    fn ambient_construction_does_not_affect_existing(
        left in any::<u64>(),
        right in any::<u64>(),
        bits in 1_usize..=128,
    ) {
        let left_value = uint(left);
        let right_value = uint(right);
        PrecisionContext::with_bounded(bits, || {
            let sum = &left_value + &right_value;
            prop_assert_eq!(sum.precision, Precision::Unlimited);
            prop_assert_eq!(sum.to_u128(), Some(u128::from(left) + u128::from(right)));
            Ok(())
        })?;
    }

    #[test]
    fn precision_context_with_bounded_works(bits in 1_usize..=256) {
        PrecisionContext::with_bounded(bits, || {
            prop_assert_eq!(PrecisionContext::active(), AmbientPrecision::Bounded(nz(bits)));
            Ok(())
        })?;
    }
}
