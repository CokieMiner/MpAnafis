//! Properties for optional `num-traits` integrations.

#[cfg(feature = "num-traits")]
use super::*;

#[cfg(feature = "num-traits")]
proptest! {
    #[test]
    fn num_from_str_radix_round_trip(
        value in strategies::uint(4),
        radix in 2_u32..=36,
    ) {
        let encoded = value.to_string_radix(radix);
        let parsed = <MpUint as ::num_traits::Num>::from_str_radix(&encoded, radix)
            .expect("value encoded in the same valid radix");
        prop_assert_eq!(parsed, value);
    }

    #[test]
    fn num_signed_trait(positive_value in 0_i64..=1_000_i64, negative_magnitude in 1_i64..=1_000_i64) {
        let positive = int_from_i64(positive_value);
        prop_assert!(positive.is_positive() || positive.is_zero());
        prop_assert!(!positive.is_negative());

        let negative = int_from_i64(negative_magnitude.wrapping_neg());
        prop_assert!(negative.is_negative());
        prop_assert!(!negative.is_positive());

        if positive.is_zero() {
            prop_assert_eq!(positive.signum().to_i64(), Some(0));
        } else {
            prop_assert_eq!(positive.signum().to_i64(), Some(1));
        }
        prop_assert_eq!(negative.signum().to_i64(), Some(-1));
    }

    #[test]
    fn num_to_primitive_mpuint(value in any::<u64>()) {
        let uint_value = uint(value);
        prop_assert_eq!(uint_value.to_u64(), Some(value));
        prop_assert_eq!(uint_value.to_u128(), Some(u128::from(value)));
        prop_assert_eq!(uint_value.to_i64(), i64::try_from(value).ok());
        prop_assert_eq!(uint_value.to_isize(), isize::try_from(value).ok());
    }

    #[test]
    fn num_to_primitive_mpint(value in any::<i64>()) {
        let int_value = int_from_i64(value);
        prop_assert_eq!(int_value.to_i64(), Some(value));
        prop_assert_eq!(int_value.to_i128(), Some(i128::from(value)));
        if value >= 0 {
            prop_assert_eq!(int_value.to_u64(), u64::try_from(value).ok());
        } else {
            prop_assert!(int_value.to_u64().is_none());
        }
    }

    #[test]
    fn num_from_primitive_mpuint(value in any::<u64>()) {
        use ::num_traits::Unsigned;
        fn require_unsigned<T: Unsigned>() {}

        require_unsigned::<MpUint>();
        prop_assert!(MpUint::zero().value.is_zero());
        prop_assert!(MpUint::one().value.is_one());
        prop_assert!(MpInt::zero().value.abs.is_zero());
        prop_assert!(MpInt::one().value.abs.is_one());

        let uint_value = <MpUint as ::num_traits::FromPrimitive>::from_u64(value)
            .expect("unsigned primitive always fits");
        prop_assert_eq!(uint_value.to_u64(), Some(value));
        prop_assert!(
            <MpUint as ::num_traits::FromPrimitive>::from_i64(-1).is_none()
        );
    }

    #[test]
    fn num_from_primitive_mpint(value in any::<i64>(), unsigned_value in any::<u64>()) {
        let int_value = <MpInt as ::num_traits::FromPrimitive>::from_i64(value)
            .expect("signed primitive always fits");
        prop_assert_eq!(int_value.to_i64(), Some(value));

        let from_unsigned = <MpInt as ::num_traits::FromPrimitive>::from_u64(unsigned_value)
            .expect("unsigned primitive always fits");
        prop_assert!(from_unsigned.is_positive() || from_unsigned.is_zero());
    }

    #[test]
    fn num_traits_roundtrip(unsigned_value in any::<u64>(), signed_value in any::<i64>(), small in any::<u8>()) {
        let uint_value = <MpUint as ::num_traits::FromPrimitive>::from_u64(unsigned_value)
            .expect("unsigned primitive always fits");
        prop_assert_eq!(uint_value.to_u64(), Some(unsigned_value));

        let int_value = <MpInt as ::num_traits::FromPrimitive>::from_i64(signed_value)
            .expect("signed primitive always fits");
        prop_assert_eq!(int_value.to_i64(), Some(signed_value));

        let small_value = uint(u64::from(small));
        prop_assert_eq!(u8::try_from(small_value).expect("fits"), small);
    }

    #[test]
    fn prop_num_traits_properties(n in -1000_i64..=1000_i64) {
        use ::num_traits::{FromPrimitive, One, ToPrimitive, Zero};
        prop_assert!(<MpUint as Zero>::zero().is_zero());
        prop_assert!(<MpInt as One>::one().is_one());
        let parsed = <MpUint as ::num_traits::Num>::from_str_radix("100", 16)
            .expect("should succeed");
        prop_assert_eq!(parsed, MpUint::from(256_u32));
        if let Some(from_prim) = <MpInt as FromPrimitive>::from_i64(n) {
            prop_assert_eq!(<MpInt as ToPrimitive>::to_i64(&from_prim), Some(n));
        }
    }
}

#[cfg(all(feature = "num-traits", feature = "std"))]
proptest! {
    #[test]
    fn mpint_from_unsigned_primitives_widens_at_signed_max(
        u64_bits in 1_usize..=64,
        u128_bits in 1_usize..=128,
        usize_bits in 1_usize..=usize::BITS as usize,
    ) {
        macro_rules! assert_ambient_boundary {
            ($ambient_bits:expr, $value:expr, $expected_bits:expr, $from_method:ident, $to_method:ident) => {{
                PrecisionContext::with_bounded($ambient_bits, || {
                    let from_value = MpInt::from($value);
                    let from_trait =
                        <MpInt as ::num_traits::FromPrimitive>::$from_method($value)
                            .expect("unsigned primitive fits");

                    prop_assert_eq!(from_value.$to_method(), Some($value));
                    prop_assert_eq!(from_trait.$to_method(), Some($value));
                    prop_assert_eq!(
                        from_value.precision,
                        Precision::Bounded(nz($expected_bits))
                    );
                    prop_assert_eq!(from_trait.precision, from_value.precision);
                    Ok(())
                })?;
            }};
        }

        let u64_signed_max = (1_u64 << (u64_bits - 1)) - 1;
        assert_ambient_boundary!(u64_bits, u64_signed_max, u64_bits, from_u64, to_u64);
        assert_ambient_boundary!(
            u64_bits,
            u64_signed_max + 1,
            u64_bits + 1,
            from_u64,
            to_u64
        );

        let u128_signed_max = (1_u128 << (u128_bits - 1)) - 1;
        assert_ambient_boundary!(
            u128_bits,
            u128_signed_max,
            u128_bits,
            from_u128,
            to_u128
        );
        assert_ambient_boundary!(
            u128_bits,
            u128_signed_max + 1,
            u128_bits + 1,
            from_u128,
            to_u128
        );

        let usize_signed_max = (1_usize << (usize_bits - 1)) - 1;
        assert_ambient_boundary!(
            usize_bits,
            usize_signed_max,
            usize_bits,
            from_usize,
            to_usize
        );
        assert_ambient_boundary!(
            usize_bits,
            usize_signed_max + 1,
            usize_bits + 1,
            from_usize,
            to_usize
        );
    }
}
