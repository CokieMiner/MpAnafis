//! Large-input multiplication properties exercised through the public API.

use proptest::prelude::{
    Just, ProptestConfig, prop_assert, prop_assert_eq, prop_assume, prop_oneof, proptest,
};

use super::support::{exact_limb_vec, uint_from_words};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn prop_dispatch_mul_div_round_trip(
        (limb_count, limbs_a, limbs_b) in prop_oneof![
            (Just(16_usize), exact_limb_vec(16), exact_limb_vec(16)),
            (Just(32_usize), exact_limb_vec(32), exact_limb_vec(32)),
            (Just(64_usize), exact_limb_vec(64), exact_limb_vec(64)),
            (Just(128_usize), exact_limb_vec(128), exact_limb_vec(128)),
        ],
    ) {
        let left_value = uint_from_words(&limbs_a);
        let right_value = uint_from_words(&limbs_b);
        prop_assume!(!left_value.is_zero());
        let product = &left_value * &right_value;
        prop_assert!(
            !product.is_zero() || right_value.is_zero(),
            "product of non-zero inputs should be non-zero"
        );
        let recovered = &product / &left_value;
        prop_assert_eq!(
            recovered,
            right_value,
            "product / a should equal b at {} limbs",
            limb_count
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1))]

    #[test]
    #[cfg_attr(
        miri,
        ignore = "Very large multiplication and division take too long under the Miri interpreter"
    )]
    fn prop_very_large_mul_div_round_trip(
        limbs_a in exact_limb_vec(10_240),
        limbs_b in exact_limb_vec(10_240),
    ) {
        let left_value = uint_from_words(&limbs_a);
        let right_value = uint_from_words(&limbs_b);
        prop_assume!(!left_value.is_zero());
        let product = &left_value * &right_value;
        prop_assert!(
            !product.is_zero() || right_value.is_zero(),
            "product of non-zero inputs should be non-zero"
        );
        let recovered = &product / &left_value;
        prop_assert_eq!(
            recovered,
            right_value,
            "product / a should equal b at 10240 limbs"
        );
    }
}
