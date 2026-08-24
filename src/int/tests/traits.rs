//! Property tests for core and optional ecosystem trait implementations.

use alloc::vec::Vec;

use proptest::prelude::*;

#[cfg(feature = "num-traits")]
use super::strategies;
use crate::int::api::{MpInt, MpUint, Precision};

proptest! {
    #[test]
    fn prop_core_iterator_aggregates(
        unsigned_values in proptest::collection::vec(0_u16..=20, 0..=8),
        signed_values in proptest::collection::vec(-10_i16..=10, 0..=8),
    ) {
        let unsigned_inputs: Vec<_> = unsigned_values
            .iter()
            .copied()
            .map(|value| {
                let mut integer = MpUint::from(value);
                integer.precision = Precision::Unlimited;
                integer
            })
            .collect();
        let unsigned_sum: MpUint = unsigned_inputs.iter().cloned().sum();
        let unsigned_product: MpUint = unsigned_inputs.iter().cloned().product();
        let expected_unsigned_sum = unsigned_values
            .iter()
            .copied()
            .map(u64::from)
            .fold(0_u64, u64::wrapping_add);
        let expected_unsigned_product = unsigned_values
            .iter()
            .copied()
            .map(u64::from)
            .fold(1_u64, u64::wrapping_mul);
        prop_assert_eq!(unsigned_sum.to_u64(), Some(expected_unsigned_sum));
        prop_assert_eq!(unsigned_product.to_u64(), Some(expected_unsigned_product));

        let signed_inputs: Vec<_> = signed_values
            .iter()
            .copied()
            .map(|value| {
                let mut integer = MpInt::from(value);
                integer.precision = Precision::Unlimited;
                integer
            })
            .collect();
        let signed_sum: MpInt = signed_inputs.iter().cloned().sum();
        let signed_product: MpInt = signed_inputs.iter().cloned().product();
        let expected_signed_sum = signed_values
            .iter()
            .copied()
            .map(i64::from)
            .fold(0_i64, i64::wrapping_add);
        let expected_signed_product = signed_values
            .iter()
            .copied()
            .map(i64::from)
            .fold(1_i64, i64::wrapping_mul);
        prop_assert_eq!(signed_sum.to_i64(), Some(expected_signed_sum));
        prop_assert_eq!(signed_product.to_i64(), Some(expected_signed_product));
    }
}

#[cfg(feature = "num-traits")]
proptest! {
    #[test]
    fn prop_num_traits_delegate_to_inherent_api(
        value in strategies::int(8),
        other in strategies::int(8),
        radix in 2_u32..=36,
    ) {
        use num_traits::{Num, Signed};

        let encoded = value.to_string_radix(radix);
        let inherent_parse = MpInt::from_str_radix(&encoded, radix).expect("generated encoding");
        let trait_parse = <MpInt as Num>::from_str_radix(&encoded, radix)
            .expect("generated encoding");
        prop_assert_eq!(trait_parse, inherent_parse);
        prop_assert_eq!(
            <MpInt as Signed>::abs_sub(&value, &other),
            value.abs_sub(&other),
        );
    }
}
