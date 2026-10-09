//! Strict prime searches across lower, native, and sieve boundaries.

use proptest::{prelude::ProptestConfig, test_runner::TestRunner};

use super::{super::InternalMpUint, native::is_prime_usize};

#[test]
fn strict_searches_match_bounded_trial_division_and_preserve_order() {
    let check = |value: usize| {
        let input = InternalMpUint::from_limb(value);
        let next = (value.checked_add(1).expect("small fixture")..=20_101)
            .find(|&candidate| is_prime_usize(candidate))
            .expect("prime above bounded fixture");
        let previous = (2..value)
            .rev()
            .find(|&candidate| is_prime_usize(candidate));
        let next_prime = input.next_prime();
        assert_eq!(next_prime, InternalMpUint::from_limb(next));
        assert!(next_prime > input);
        assert!(next_prime.is_prime());
        assert_eq!(input.prev_prime(), previous.map(InternalMpUint::from_limb));
    };
    for value in 0..=17 {
        check(value);
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 64 }))
        .run(&(0_usize..=20_000, 0_usize..=20_000), |(left, right)| {
            check(left);
            check(right);
            let smaller = InternalMpUint::from_limb(left.min(right));
            let larger = InternalMpUint::from_limb(left.max(right));
            assert!(smaller.next_prime() <= larger.next_prime());
            Ok(())
        })
        .expect("strict search property");
}

#[test]
fn searches_cross_native_widths_without_including_the_input() {
    for (start, expected) in [
        ("18446744073709551616", "18446744073709551629"),
        ("18446744073709551557", "18446744073709551629"),
        (
            "340282366920938463463374607431768211456",
            "340282366920938463463374607431768211507",
        ),
        (
            "340282366920938463463374607431768211299",
            "340282366920938463463374607431768211507",
        ),
    ] {
        let input = InternalMpUint::from_str_radix(start, 10).expect("decimal fixture");
        let prime = InternalMpUint::from_str_radix(expected, 10).expect("decimal prime");
        assert_eq!(input.next_prime(), prime);
    }
    for (start, expected) in [
        ("18446744073709551616", "18446744073709551557"),
        ("18446744073709551629", "18446744073709551557"),
        (
            "340282366920938463463374607431768211456",
            "340282366920938463463374607431768211297",
        ),
    ] {
        let input = InternalMpUint::from_str_radix(start, 10).expect("decimal fixture");
        let prime = InternalMpUint::from_str_radix(expected, 10).expect("decimal prime");
        assert_eq!(input.prev_prime(), Some(prime));
    }
}
