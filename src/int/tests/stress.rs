//! Large-input arithmetic and allocation-boundary properties.

use super::*;

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
        let left_value = ArbiUint {
            value: InternalArbiUint::from_limbs(limbs_a_seed),
            precision: Precision::Unlimited,
        };
        let right_value = ArbiUint {
            value: InternalArbiUint::from_limbs(limbs_b),
            precision: Precision::Unlimited,
        };
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
        let base_value = ArbiUint {
            value: InternalArbiUint::from_limbs(limbs_seed),
            precision: Precision::Unlimited,
        };
        let factor_a = ArbiUint::from(factor);
        let value_a = &base_value * &factor_a;
        let factor_b = ArbiUint::from(factor.wrapping_mul(3).wrapping_add(7));
        let value_b = &base_value * &factor_b;
        let gcd_value = value_a.value.gcd(&value_b.value);
        prop_assert_eq!(
            value_a
                .value
                .rem(&gcd_value),
            InternalArbiUint::zero(),
            "gcd must divide a"
        );
        prop_assert_eq!(
            value_b
                .value
                .rem(&gcd_value),
            InternalArbiUint::zero(),
            "gcd must divide b"
        );
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
        let value = ArbiUint {
            value: InternalArbiUint::from_limbs(limbs),
            precision: Precision::Unlimited,
        };
        let root = value.isqrt().expect("isqrt should succeed");
        let root_sq = &root * &root;
        prop_assert!(root_sq <= value, "isqrt^2 > a at {} limbs", limb_count);
        let next_root = &root + &ArbiUint::one();
        let next_sq = &next_root * &next_root;
        prop_assert!(value <= next_sq, "isqrt too small at {} limbs", limb_count);
    }
}

proptest! {
    #[test]
    fn stress_allocation_inline_heap_boundary(extra_bits in 0_usize..=64) {
        let target_bits = 260_usize.wrapping_add(extra_bits);
        let mut value = ArbiUint::one();
        for _ in 0..target_bits {
            value = &value * &ArbiUint::from(2_u32);
        }
        prop_assert_eq!(value.precision, Precision::Unlimited);
        prop_assert!(value.value.significant_bits() >= target_bits);
        let decimal = value.to_string();
        let roundtrip: ArbiUint = decimal.parse().expect("roundtrip");
        prop_assert_eq!(
            roundtrip,
            value,
            "serialization roundtrip across inline/heap boundary"
        );
    }
}
