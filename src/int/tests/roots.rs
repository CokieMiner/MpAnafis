//! Root maximality, square decomposition, and signed-domain contracts.

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{MpInt, MpUint, Precision};

use super::{strategies, support::nz};

proptest! {
    #[test]
    fn large_roots_are_maximal_and_square_remainders_recompose(
        unsigned in strategies::uint(8), signed in strategies::int(8), degree in 1_u32..=5,
    ) {
        let (root, remainder) = unsigned.sqrt_rem().expect("unsigned square roots are total");
        prop_assert_eq!(unsigned.isqrt(), Some(root.clone()));
        prop_assert_eq!(&(&root * &root + &remainder), &unsigned);
        prop_assert!(remainder < (&root << 1_usize) + MpUint::one());
        prop_assert!(unsigned < (&root + MpUint::one()).square());
        prop_assert_eq!(unsigned.square().isqrt(), Some(unsigned.clone()));
        prop_assert_eq!(unsigned.is_perfect_square(), remainder.is_zero());

        let magnitude = signed.unsigned_abs();
        let (signed_root, signed_remainder) = signed.sqrt_rem().expect("unlimited absolute square root");
        prop_assert_eq!(signed_root.square() + &signed_remainder, MpInt::from(magnitude.clone()));
        prop_assert_eq!(signed.checked_isqrt(), (!signed.is_negative()).then_some(signed_root));
        prop_assert_eq!(signed.square().checked_isqrt(), Some(signed.abs()));
        prop_assert_eq!(signed.is_perfect_square(), signed_remainder.is_zero());
        for value in [&unsigned, &magnitude] {
            let nth_root = value.nth_root(degree).expect("positive degree");
            prop_assert!(nth_root.pow(degree) <= *value);
            prop_assert!(*value < (&nth_root + MpUint::one()).pow(degree));
        }
        prop_assert_eq!(signed.nth_root(degree), magnitude.nth_root(degree).map(MpInt::from));
        prop_assert_eq!(unsigned.nth_root(0), None);
        prop_assert_eq!(signed.nth_root(0), None);
    }

    #[test]
    fn bounded_square_roots_match_native_and_preserve_precision(
        unsigned_seed in any::<u128>(), signed_seed in any::<i128>(), bits in 1_usize..=128,
        degree in 0_u32..=8,
    ) {
        let width = nz(bits);
        let unsigned = MpUint::with_precision_wrapping(unsigned_seed, width);
        let signed = MpInt::with_precision_wrapping(signed_seed, width);
        let u = unsigned.to_u128().expect("native width");
        let i = signed.to_i128().expect("native width");
        let native_unsigned_root = u.isqrt();
        let native_signed_root = i.unsigned_abs().isqrt();
        let (actual_root, actual_remainder) = unsigned.sqrt_rem().expect("unsigned square roots are total");
        prop_assert_eq!(actual_root.to_u128(), Some(native_unsigned_root));
        prop_assert_eq!(actual_remainder.to_u128(), Some(u - native_unsigned_root * native_unsigned_root));
        prop_assert_eq!(unsigned.isqrt(), Some(actual_root.clone()));
        prop_assert_eq!(actual_root.precision(), Precision::Bounded(width));
        prop_assert_eq!(actual_remainder.precision(), Precision::Bounded(width));
        prop_assert_eq!(signed.checked_isqrt().and_then(|value| value.to_u128()), (i >= 0).then_some(native_signed_root));
        let result = signed.sqrt_rem();
        prop_assert_eq!(result.is_none(), bits == 1 && i == -1);
        if let Some((root, remainder)) = result {
            prop_assert_eq!(root.to_u128(), Some(native_signed_root));
            prop_assert_eq!(remainder.to_u128(), Some(i.unsigned_abs() - native_signed_root * native_signed_root));
            prop_assert_eq!(root.precision(), Precision::Bounded(width));
            prop_assert_eq!(remainder.precision(), Precision::Bounded(width));
        }
        let exact_unsigned = (MpUint::zero() + &unsigned).nth_root(degree);
        let actual_unsigned = unsigned.nth_root(degree);
        prop_assert_eq!(&actual_unsigned, &exact_unsigned);
        if let Some(root) = actual_unsigned {
            prop_assert_eq!(root.precision(), Precision::Bounded(width));
        }
        let maximum = (i128::MAX >> (128 - bits)).cast_unsigned();
        let exact_signed = MpInt::from(i).nth_root(degree);
        let expected = exact_signed.filter(|root| root.to_u128().is_some_and(|value| value <= maximum));
        let actual_signed = signed.nth_root(degree);
        prop_assert_eq!(&actual_signed, &expected);
        if let Some(root) = actual_signed {
            prop_assert_eq!(root.precision(), Precision::Bounded(width));
        }
    }
}
