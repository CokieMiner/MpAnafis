//! Native prime-search references and classification of known pseudoprimes.

use proptest::prelude::{any, prop_assert_eq, proptest};

use crate::{MpInt, MpUint};

use super::support::nz;

proptest! {
    #[test]
    fn bounded_classification_and_strict_neighbors_match_trial_division(
        unsigned_seed in any::<u16>(), signed_seed in any::<i16>(), bits in 1_usize..=16,
    ) {
        let prime = |value: u64| value >= 2 && (2..=value.isqrt()).all(|divisor| !value.is_multiple_of(divisor));
        let unsigned = MpUint::with_precision_wrapping(unsigned_seed, nz(bits));
        let signed = MpInt::with_precision_wrapping(signed_seed, nz(bits));
        let u = unsigned.to_u64().expect("at most sixteen bits");
        let i = signed.to_i64().expect("at most sixteen signed bits");
        prop_assert_eq!(unsigned.is_prime(), prime(u));
        prop_assert_eq!(signed.is_prime(), i >= 2 && prime(i.cast_unsigned()));
        let unsigned_maximum = (1_u64 << bits) - 1;
        let signed_maximum = i64::MAX >> (64 - bits);
        // 65_537 is prime and exceeds every generated sixteen-bit magnitude.
        let unsigned_next = (u + 1..=65_537).find(|value| prime(*value)).expect("next native prime");
        let unsigned_previous = (2..u).rev().find(|value| prime(*value));
        let signed_next = (i.max(1).cast_unsigned() + 1..=65_537).find(|value| prime(*value)).expect("next native prime");
        let signed_previous = (2..i.max(2).cast_unsigned()).rev().find(|value| prime(*value));
        let unsigned_next_result = unsigned.next_prime();
        let unsigned_previous_result = unsigned.prev_prime();
        prop_assert_eq!(unsigned_next_result.as_ref().and_then(MpUint::to_u64), (unsigned_next <= unsigned_maximum).then_some(unsigned_next));
        prop_assert_eq!(unsigned_previous_result.as_ref().and_then(MpUint::to_u64), unsigned_previous);
        for result in [unsigned_next_result, unsigned_previous_result].into_iter().flatten() {
            prop_assert_eq!(result.precision(), unsigned.precision());
        }
        let signed_next_result = signed.next_prime();
        let signed_previous_result = signed.prev_prime();
        prop_assert_eq!(signed_next_result.as_ref().and_then(MpInt::to_u64), (signed_next <= signed_maximum.cast_unsigned()).then_some(signed_next));
        prop_assert_eq!(signed_previous_result.as_ref().and_then(MpInt::to_u64), signed_previous);
        for result in [signed_next_result, signed_previous_result].into_iter().flatten() {
            prop_assert_eq!(result.precision(), signed.precision());
        }
    }
}

#[test]
fn known_mersenne_primes_and_carmichael_composites_are_classified() {
    let primes = [
        2_u8, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97,
    ];
    for value in 0_u8..=100 {
        assert_eq!(MpUint::from(value).is_prime(), primes.contains(&value));
        assert_eq!(MpInt::from(value).is_prime(), primes.contains(&value));
    }
    for value in [-13_i8, -3, -1] {
        assert!(!MpInt::from(value).is_prime());
        assert_eq!(MpInt::from(value).next_prime(), Some(MpInt::from(2_u8)));
        assert_eq!(MpInt::from(value).prev_prime(), None);
    }
    for exponent in [17_usize, 127] {
        let mersenne = (MpUint::one() << exponent) - MpUint::one();
        assert!(mersenne.is_prime());
    }
    for value in [561_u32, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841] {
        assert!(!MpUint::from(value).is_prime());
    }
}
