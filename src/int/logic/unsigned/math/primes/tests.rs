//! Properties and regressions for primality operations.

use alloc::vec;

use proptest::prelude::*;

use super::*;

fn is_prime_usize(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n & 1 == 0 {
        return false;
    }
    let mut divisor = 3_usize;
    while divisor.saturating_mul(divisor) <= n {
        if n.checked_rem(divisor) == Some(0) {
            return false;
        }
        divisor = divisor.wrapping_add(2);
    }
    true
}

fn next_prime_usize(mut n: usize) -> usize {
    if n < 2 {
        return 2;
    }
    if n & 1 == 0 {
        n = n.wrapping_add(1);
    }
    while !is_prime_usize(n) {
        n = n.wrapping_add(2);
    }
    n
}

proptest! {
    #[test]
    fn prop_is_prime_matches_bruteforce_for_small_inputs(value in 0_usize..=50_000_usize) {
        let internal_value = InternalMpUint::from_limb(value);
        prop_assert_eq!(internal_value.is_prime(), is_prime_usize(value));
    }

    #[test]
    fn prop_is_probably_prime_accepts_small_primes(value in prop_oneof![
        Just(2_usize), Just(3), Just(5), Just(7), Just(11), Just(13), Just(17), Just(19),
        Just(23), Just(29), Just(31), Just(37), Just(41), Just(43), Just(47), Just(53),
        Just(59), Just(61), Just(67), Just(71), Just(73), Just(79), Just(83), Just(89),
        Just(97), Just(101), Just(103), Just(107), Just(109), Just(113), Just(127),
        Just(131), Just(137), Just(139), Just(149), Just(151), Just(157), Just(163),
        Just(167), Just(173), Just(179), Just(181), Just(191), Just(193), Just(197),
        Just(199), Just(211), Just(223), Just(227), Just(229), Just(233), Just(239),
        Just(241), Just(251), Just(257), Just(263), Just(269), Just(271), Just(277),
        Just(281), Just(283), Just(293), Just(307), Just(311)
    ]) {
        let internal_value = InternalMpUint::from_limb(value);
        prop_assert!(internal_value.is_probably_prime(24));
    }

    /// The trial-division screen added to `is_probably_prime` only runs for
    /// inputs too wide for the deterministic u64 test, so it needs multi-limb
    /// operands to be exercised at all. Multiplying a wide odd cofactor by a
    /// screened prime produces a composite the screen must reject without
    /// reaching Miller-Rabin.
    #[test]
    fn prop_is_probably_prime_rejects_wide_small_prime_multiples(
        cofactor_limbs in proptest::collection::vec(any::<Limb>(), 3..=5),
        prime_index in 0_usize..SIEVE_PRIMES.len(),
    ) {
        let prime = SIEVE_PRIMES.get(prime_index).copied().unwrap_or(3);
        let cofactor = InternalMpUint::from_limbs(cofactor_limbs);
        let composite = cofactor.mul(&InternalMpUint::from_limb(prime));

        // A cofactor of 0 or 1 leaves the product prime or zero, and the
        // product must exceed u64 for the wide path to run at all.
        if composite.to_u64().is_none() {
            prop_assert!(!composite.is_probably_prime(24));
            prop_assert!(!composite.is_prime());
        }
    }

    /// The screen must not reject primes. These are the smallest primes above
    /// 2^64 and 2^128, so both take the wide path.
    #[test]
    fn prop_wide_primes_survive_the_screen(select in 0_usize..2) {
        let decimal = if select == 0 {
            "18446744073709551629"
        } else {
            "340282366920938463463374607431768211507"
        };
        let prime = InternalMpUint::from_str_radix(decimal, 10)
            .expect("literal is a valid decimal string");
        prop_assert!(prime.is_probably_prime(24));
        prop_assert!(prime.is_prime());
    }

    #[test]
    fn prop_next_prime_matches_bruteforce_small_inputs(value in 0_usize..=20_000_usize) {
        let internal_value = InternalMpUint::from_limb(value);
        let next_prime = internal_value.next_prime();
        let expected_prime = next_prime_usize(value);
        prop_assert_eq!(&next_prime, &InternalMpUint::from_limb(expected_prime));
        prop_assert!(next_prime.is_prime());
        prop_assert!(next_prime.cmp(&internal_value) != Ordering::Less);
    }

    #[test]
    fn prop_next_prime_output_is_prime_and_monotone(
        left_value in 0_usize..=10_000_usize,
        right_value in 0_usize..=10_000_usize,
    ) {
        let left_internal = InternalMpUint::from_limb(left_value);
        let right_internal = InternalMpUint::from_limb(right_value);
        let left_prime = left_internal.next_prime();
        let right_prime = right_internal.next_prime();

        prop_assert!(left_prime.is_prime());
        prop_assert!(right_prime.is_prime());
        if left_value <= right_value {
            prop_assert!(left_prime.cmp(&right_prime) != Ordering::Greater);
        }
    }
}
