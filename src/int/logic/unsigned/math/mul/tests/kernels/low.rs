//! Property tests for truncated low-product multiplication.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;

proptest! {
    #[test]
    #[allow(
        clippy::indexing_slicing,
        reason = "Property validates exact low-product slice boundaries"
    )]
    fn prop_mullo_slice_matches_full_mul(
        len in prop_oneof![Just(1_usize), Just(2), Just(5), Just(15), Just(31), Just(32), Just(33), Just(50), Just(100)],
        a_seed in proptest::collection::vec(any::<Limb>(), 100),
        b_seed in proptest::collection::vec(any::<Limb>(), 100),
    ) {
        let mut scratch = MulScratch::default();
        let a = a_seed[..len].to_vec();
        let b = b_seed[..len].to_vec();
        let mut dst = vec![0; len];
        let mut full_dst = vec![0; len.wrapping_mul(2)];
        Schoolbook::mul(&mut full_dst, &a, &b);
        LowProduct::mul(&mut dst, &a, &b, len, &mut scratch);
        prop_assert_eq!(&dst[..], &full_dst[..len], "mullo_slice failed for len={}", len);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    #[allow(
        clippy::indexing_slicing,
        reason = "Property fixes seed vector length above every selected low-product length"
    )]
    fn prop_mullo_slice_matches_full_mul_large(
        len in prop_oneof![Just(127_usize), Just(128), Just(255), Just(256), Just(511), Just(640)],
        a_seed in proptest::collection::vec(any::<Limb>(), 640),
        b_seed in proptest::collection::vec(any::<Limb>(), 640),
    ) {
        let mut scratch = MulScratch::default();
        let random_a = a_seed[..len].to_vec();
        let random_b = b_seed[..len].to_vec();
        let all_max = vec![Limb::MAX; len];
        let near_max_a: Vec<Limb> = random_a
            .iter()
            .map(|limb| Limb::MAX.wrapping_sub(*limb & Limb::from(u8::MAX)))
            .collect();
        let near_max_b: Vec<Limb> = random_b
            .iter()
            .map(|limb| Limb::MAX.wrapping_sub(*limb & Limb::from(u8::MAX)))
            .collect();

        for (case, a, b) in [
            ("random", &random_a, &random_b),
            ("all-max", &all_max, &all_max),
            ("near-max", &near_max_a, &near_max_b),
        ] {
            let mut dst = vec![0; len];
            let mut full_dst = vec![0; len.wrapping_mul(2)];
            Schoolbook::mul(&mut full_dst, a, b);
            LowProduct::mul(&mut dst, a, b, len, &mut scratch);
            prop_assert_eq!(
                &dst[..],
                &full_dst[..len],
                "large mullo_slice failed for case={}, len={}",
                case,
                len
            );
        }

        let mut square_dst = vec![0; len];
        let mut full_square = vec![0; len.wrapping_mul(2)];
        Schoolbook::mul(&mut full_square, &random_a, &random_a);
        LowProduct::mul(&mut square_dst, &random_a, &random_a, len, &mut scratch);
        prop_assert_eq!(
            &square_dst[..],
            &full_square[..len],
            "aliased-input mullo square failed for len={}",
            len
        );
    }
}
