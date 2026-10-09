//! Property tests for core and optional ecosystem trait implementations.

use alloc::vec::Vec;

use proptest::prelude::{prop_assert_eq, proptest};

use crate::{MpInt, MpUint};

proptest! {
    #[test]
    fn prop_core_iterator_aggregates(
        unsigned_values in proptest::collection::vec(0_u16..=20, 0..=8),
        signed_values in proptest::collection::vec(-10_i16..=10, 0..=8),
    ) {
        let unsigned_inputs: Vec<_> = unsigned_values
            .iter()
            .copied()
            .map(|value| MpUint::zero() + MpUint::from(value))
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
        prop_assert_eq!(unsigned_inputs.iter().sum::<MpUint>().to_u64(), Some(expected_unsigned_sum));
        prop_assert_eq!(unsigned_inputs.iter().product::<MpUint>().to_u64(), Some(expected_unsigned_product));

        let signed_inputs: Vec<_> = signed_values
            .iter()
            .copied()
            .map(|value| MpInt::zero() + MpInt::from(value))
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
        prop_assert_eq!(signed_inputs.iter().sum::<MpInt>().to_i64(), Some(expected_signed_sum));
        prop_assert_eq!(signed_inputs.iter().product::<MpInt>().to_i64(), Some(expected_signed_product));
    }
}
