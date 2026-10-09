//! Properties for optional `num-traits` integrations.

use num_traits::{FromPrimitive, Num, One, Signed, ToPrimitive, Unsigned, Zero};
use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint};
#[cfg(feature = "std")]
use crate::{Precision, PrecisionContext};

#[cfg(feature = "std")]
use super::support::nz;

proptest! {
    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Native integer-to-float casts provide independent rounded conversion references"
    )]
    fn numeric_traits_match_primitive_domains_and_signed_arithmetic(
        unsigned in any::<u128>(), signed in any::<i128>(),
        native_unsigned in any::<usize>(), native_signed in any::<isize>(),
        radix in 2_u32..=36,
    ) {
        fn require_unsigned<T: Unsigned>() {}
        require_unsigned::<MpUint>();
        let u = <MpUint as FromPrimitive>::from_u128(unsigned).expect("unsigned primitive fits");
        let i = <MpInt as FromPrimitive>::from_i128(signed).expect("signed primitive fits");
        let positive = <MpInt as FromPrimitive>::from_u128(unsigned).expect("unsigned magnitude fits");
        prop_assert_eq!(<MpUint as ToPrimitive>::to_u128(&u), Some(unsigned));
        prop_assert_eq!(<MpInt as ToPrimitive>::to_i128(&i), Some(signed));
        prop_assert_eq!(<MpInt as ToPrimitive>::to_u128(&positive), Some(unsigned));
        prop_assert_eq!(<MpUint as ToPrimitive>::to_u64(&u), u64::try_from(unsigned).ok());
        prop_assert_eq!(<MpUint as ToPrimitive>::to_i64(&u), i64::try_from(unsigned).ok());
        prop_assert_eq!(<MpUint as ToPrimitive>::to_i128(&u), i128::try_from(unsigned).ok());
        prop_assert_eq!(<MpUint as ToPrimitive>::to_usize(&u), usize::try_from(unsigned).ok());
        prop_assert_eq!(<MpUint as ToPrimitive>::to_isize(&u), isize::try_from(unsigned).ok());
        prop_assert_eq!(<MpInt as ToPrimitive>::to_u64(&i), u64::try_from(signed).ok());
        prop_assert_eq!(<MpInt as ToPrimitive>::to_u128(&i), u128::try_from(signed).ok());
        prop_assert_eq!(<MpInt as ToPrimitive>::to_i64(&i), i64::try_from(signed).ok());
        prop_assert_eq!(<MpInt as ToPrimitive>::to_usize(&i), usize::try_from(signed).ok());
        prop_assert_eq!(<MpInt as ToPrimitive>::to_isize(&i), isize::try_from(signed).ok());
        prop_assert_eq!(<MpUint as FromPrimitive>::from_i128(signed).and_then(|value| value.to_u128()), u128::try_from(signed).ok());
        prop_assert_eq!(<MpUint as FromPrimitive>::from_usize(native_unsigned).and_then(|value| value.to_usize()), Some(native_unsigned));
        prop_assert_eq!(<MpInt as FromPrimitive>::from_isize(native_signed).and_then(|value| value.to_isize()), Some(native_signed));
        prop_assert_eq!(<MpInt as FromPrimitive>::from_usize(native_unsigned).and_then(|value| value.to_usize()), Some(native_unsigned));
        prop_assert_eq!(<MpUint as FromPrimitive>::from_isize(native_signed).and_then(|value| value.to_isize()), (native_signed >= 0).then_some(native_signed));
        prop_assert_eq!(<MpInt as Signed>::is_positive(&i), signed > 0);
        prop_assert_eq!(<MpInt as Signed>::is_negative(&i), signed < 0);
        prop_assert_eq!(<MpInt as Signed>::signum(&i).to_i128(), Some(signed.signum()));
        prop_assert_eq!(<MpInt as Signed>::abs(&i).to_u128(), Some(signed.unsigned_abs()));
        let other = MpInt::from(native_signed);
        prop_assert_eq!(<MpInt as Signed>::abs_sub(&i, &other), if i <= other { MpInt::zero() } else { &i - &other });
        prop_assert_eq!(<MpUint as Num>::from_str_radix(&u.to_string_radix(radix), radix).expect("valid encoding"), u);
        prop_assert_eq!(<MpInt as Num>::from_str_radix(&i.to_string_radix(radix), radix).expect("valid encoding"), i);
        let uint_zero = <MpUint as Zero>::zero();
        let int_zero = <MpInt as Zero>::zero();
        prop_assert!(<MpUint as Zero>::is_zero(&uint_zero));
        prop_assert!(<MpInt as Zero>::is_zero(&int_zero));
        prop_assert!(uint_zero.to_le_bytes().is_empty());
        prop_assert!(int_zero.to_le_bytes().is_empty() && !int_zero.is_negative());
        prop_assert!(<MpUint as One>::is_one(&<MpUint as One>::one()));
        prop_assert!(<MpInt as One>::is_one(&<MpInt as One>::one()));
        for native in [0_u64, 1, u64::MAX, native_unsigned as u64] {
            let value = <MpUint as FromPrimitive>::from_u64(native).expect("fits");
            prop_assert_eq!(<MpUint as ToPrimitive>::to_f32(&value), Some(native as f32));
            prop_assert_eq!(<MpUint as ToPrimitive>::to_f64(&value), Some(native as f64));
        }
        for native in [0_i64, -1, i64::MIN, i64::MAX, native_signed as i64] {
            let value = <MpInt as FromPrimitive>::from_i64(native).expect("fits");
            prop_assert_eq!(<MpInt as ToPrimitive>::to_f32(&value), Some(native as f32));
            prop_assert_eq!(<MpInt as ToPrimitive>::to_f64(&value), Some(native as f64));
        }
    }
}

#[cfg(feature = "std")]
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
                        from_value.precision(),
                        Precision::Bounded(nz($expected_bits))
                    );
                    prop_assert_eq!(from_trait.precision(), from_value.precision());
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
