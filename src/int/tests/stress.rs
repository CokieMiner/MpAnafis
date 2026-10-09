//! Large-input arithmetic and allocation-boundary properties.

use alloc::{string::ToString, vec::Vec};

use proptest::prelude::{Just, prop_assert, prop_assert_eq, prop_oneof, proptest};

use crate::{MpUint, Precision};

use super::support::{exact_limb_vec, uint_from_words};

proptest! {
    #[test]
    fn stress_add_sub_large(
        (limb_count, limbs_a_seed, limbs_b_seed) in prop_oneof![
            (Just(32_usize), exact_limb_vec(32), exact_limb_vec(32)),
            (Just(64_usize), exact_limb_vec(64), exact_limb_vec(64)),
            (Just(128_usize), exact_limb_vec(128), exact_limb_vec(128)),
        ],
    ) {
        let limbs_b = limbs_b_seed
            .iter()
            .map(|limb| limb >> 1)
            .collect::<Vec<_>>();
        let left_value = uint_from_words(&limbs_a_seed);
        let right_value = uint_from_words(&limbs_b);
        let sum = &left_value + &right_value;
        let recovered = &sum - &right_value;
        prop_assert_eq!(recovered, left_value, "add/sub roundtrip at {} limbs", limb_count);
    }

    #[test]
    fn stress_gcd_large(
        (_limb_count, limbs_seed) in prop_oneof![
            (Just(16_usize), exact_limb_vec(16)),
            (Just(32_usize), exact_limb_vec(32)),
        ],
        factor in 2_u64..=1_000_000_u64,
    ) {
        let base_value = uint_from_words(&limbs_seed);
        let factor_a = MpUint::from(factor);
        let value_a = &base_value * &factor_a;
        let factor_b = MpUint::from(factor * 3 + 7);
        let value_b = &base_value * &factor_b;
        let gcd_value = value_a.gcd(&value_b);
        if gcd_value.is_zero() {
            prop_assert!(base_value.is_zero());
        } else {
            prop_assert!((&value_a % &gcd_value).is_zero(), "gcd must divide a");
            prop_assert!((&value_b % &gcd_value).is_zero(), "gcd must divide b");
        }
    }

    #[test]
    fn stress_isqrt_large(
        (limb_count, limbs_seed) in prop_oneof![
            (Just(8_usize), exact_limb_vec(8)),
            (Just(16_usize), exact_limb_vec(16)),
            (Just(32_usize), exact_limb_vec(32)),
        ],
    ) {
        let mut limbs = limbs_seed;
        if let Some(last_limb) = limbs.last_mut() {
            *last_limb >>= 1;
        }
        let value = uint_from_words(&limbs);
        let root = value.isqrt().expect("isqrt should succeed");
        let root_sq = &root * &root;
        prop_assert!(root_sq <= value, "isqrt^2 > a at {} limbs", limb_count);
        let next_root = &root + &MpUint::one();
        let next_sq = &next_root * &next_root;
        prop_assert!(value < next_sq, "isqrt too small at {} limbs", limb_count);
    }
}

proptest! {
    #[test]
    fn stress_allocation_inline_heap_boundary(extra_bits in 0_usize..=64) {
        let target_bits = 4 * usize::BITS as usize + extra_bits;
        let mut value = MpUint::one();
        for _ in 0..target_bits {
            value = &value * &MpUint::from(2_u32);
        }
        prop_assert_eq!(value.precision(), Precision::Unlimited);
        prop_assert_eq!(value.significant_bits(), target_bits + 1);
        let decimal = value.to_string();
        let roundtrip: MpUint = decimal.parse().expect("roundtrip");
        prop_assert_eq!(
            roundtrip,
            value,
            "serialization roundtrip across inline/heap boundary"
        );
    }
}
