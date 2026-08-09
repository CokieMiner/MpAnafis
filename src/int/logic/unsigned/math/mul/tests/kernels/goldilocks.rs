//! Property tests for Goldilocks field arithmetic and roots of unity.

use proptest::prelude::*;

use super::*;

proptest! {
    #[test]
    fn prop_goldilocks_mul_matches_u128_remainder(a in 0_u64..PRIME, b in 0_u64..PRIME) {
        let expected = u64::try_from(
            u128::from(a)
                .wrapping_mul(u128::from(b))
                .rem_euclid(PRIME_U128),
        )
        .expect("the Goldilocks remainder is strictly below 2^64");
        prop_assert_eq!(mul_mod(a, b), expected);
    }

    #[test]
    fn prop_goldilocks_roots_have_exact_power_of_two_order(exponent in 1_u32..=26) {
        let transform_len = 1_u64.wrapping_shl(exponent);
        let root = pow_mod(
            PRIMITIVE_ROOT,
            PRIME.wrapping_sub(1).div_euclid(transform_len),
        );
        prop_assert_eq!(pow_mod(root, transform_len), 1);
        prop_assert_ne!(pow_mod(root, transform_len >> 1), 1);
    }
}
