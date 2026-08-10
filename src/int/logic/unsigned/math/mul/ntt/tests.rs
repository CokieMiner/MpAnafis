//! Property tests for exact NTT/CRT multiplication.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::mul::Schoolbook;

fn operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    prop_oneof![
        (
            proptest::collection::vec(any::<Limb>(), 1),
            proptest::collection::vec(any::<Limb>(), 1),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 17),
            proptest::collection::vec(any::<Limb>(), 13),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 65),
            proptest::collection::vec(any::<Limb>(), 65),
        ),
        (
            proptest::collection::vec(any::<Limb>(), 257),
            proptest::collection::vec(any::<Limb>(), 193),
        ),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_ntt_matches_basecase((a, b) in operands()) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        prop_assert!(Ntt::try_mul(&mut actual, &a, &b));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_two_prime_digit_widths_match_basecase(
        digit_bits in 15_u32..=20,
        a in proptest::collection::vec(any::<Limb>(), 1..=65),
        b in proptest::collection::vec(any::<Limb>(), 1..=65),
    ) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let plan = TransformPlan {
            digit_bits,
            modulus_count: 2,
        };
        prop_assert!(Ntt::try_mul_with_plan(&mut actual, &a, &b, plan));
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_goldilocks_digit_widths_match_basecase(
        digit_bits in 15_u32..=23,
        a in proptest::collection::vec(any::<Limb>(), 1..=65),
        b in proptest::collection::vec(any::<Limb>(), 1..=65),
    ) {
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);
        let plan = TransformPlan {
            digit_bits,
            modulus_count: 1,
        };
        prop_assert!(Ntt::try_mul_with_plan(&mut actual, &a, &b, plan));
        prop_assert_eq!(actual, expected);
    }
}
