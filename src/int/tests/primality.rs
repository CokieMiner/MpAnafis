//! Prime and composite classification properties.

use super::*;

proptest! {
    #[test]
    fn stress_mersenne_primes(exponent in prop_oneof![Just(17_usize), Just(127_usize)]) {
        let mersenne = (MpUint::one() << exponent) - MpUint::one();
        prop_assert!(mersenne.is_prime(), "M{} should be prime", exponent);
    }

    #[test]
    fn stress_carmichael_composites(value in prop_oneof![
        Just(561_u32),
        Just(1105_u32),
        Just(1729_u32),
        Just(2465_u32),
        Just(2821_u32),
        Just(6601_u32),
        Just(8911_u32),
        Just(10585_u32),
        Just(15841_u32),
    ]) {
        let number = MpUint::from(value);
        prop_assert!(!number.is_prime(), "{} is a Carmichael number and composite", value);
    }
}

proptest! {
    #[test]
    fn prop_small_primality(n in 2_u64..=100) {
        let known_primes: &[u64] = &[
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        ];
        prop_assert_eq!(MpUint::from(n).is_prime(), known_primes.contains(&n), "{} primality mismatch", n);
    }
}
