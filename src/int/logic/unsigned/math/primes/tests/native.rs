//! Native classification and Lucas pseudoprime boundaries.

use proptest::{prelude::ProptestConfig, test_runner::TestRunner};

use crate::int::logic::unsigned::math::primes::baillie_psw::strong_lucas_selfridge;

use super::super::{InternalMpUint, MontgomeryDomain, MontgomeryScratch, ODD_COMPOSITE, Primality};

#[test]
fn primality_methods_agree_with_trial_division_for_bounded_inputs() {
    for value in 0..=311 {
        check_classification(value);
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 64 }))
        .run(&(0_usize..=50_000), |value| {
            check_classification(value);
            Ok(())
        })
        .expect("native primality property");
}

fn check_classification(value: usize) {
    let input = InternalMpUint::from_limb(value);
    let expected = is_prime_usize(value);
    assert_eq!(input.is_prime(), expected);
    assert_eq!(Primality::is_prime_limb(value), expected);
    for rounds in [0, 1, 24, 64, u32::MAX] {
        assert_eq!(input.is_probably_prime(rounds), expected);
    }
    assert_eq!(Primality::trial_division(&input), expected);
    if value >= 3 && value & 1 != 0 {
        assert_eq!(Primality::baillie_psw(&input), expected);
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Exhaustive bitmap and 20001-input Baillie-PSW comparisons require native execution; bounded classification properties run under Miri."
)]
fn bitmap_and_combined_witnesses_cover_their_complete_small_domains() {
    let maximum = ((ODD_COMPOSITE.len() - 1) << 4) | 15;
    for value in 0..=maximum.saturating_add(64) {
        assert_eq!(
            Primality::is_prime_limb(value),
            is_prime_usize(value),
            "n={value}"
        );
    }
    for value in 0..=5_000 {
        assert_eq!(
            Primality::trial_division(&InternalMpUint::from_limb(value)),
            is_prime_usize(value)
        );
    }
    for value in (3..=20_001).step_by(2) {
        assert_eq!(
            Primality::baillie_psw(&InternalMpUint::from_limb(value)),
            is_prime_usize(value)
        );
    }
}

#[test]
fn lucas_pseudoprimes_and_squares_preserve_the_combined_rejection() {
    for value in [
        5459, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519,
    ] {
        let n = InternalMpUint::from_limb(value);
        let domain = MontgomeryDomain::new::<true>(&n);
        let mut scratch = MontgomeryScratch::default();
        let mut product = InternalMpUint::zero();
        let one =
            domain.transform_into_with_scratch(&InternalMpUint::one(), &mut product, &mut scratch);
        assert!(strong_lucas_selfridge(
            &domain,
            &one,
            &mut product,
            &mut scratch
        ));
        assert!(!Primality::baillie_psw(&n));
    }
    for prime in [3, 5, 7, 251, 1009] {
        let n = InternalMpUint::from_limb(prime).square();
        let domain = MontgomeryDomain::new::<true>(&n);
        let mut scratch = MontgomeryScratch::default();
        let mut product = InternalMpUint::zero();
        let one =
            domain.transform_into_with_scratch(&InternalMpUint::one(), &mut product, &mut scratch);
        assert!(!strong_lucas_selfridge(
            &domain,
            &one,
            &mut product,
            &mut scratch
        ));
    }
}

pub fn is_prime_usize(n: usize) -> bool {
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
    while divisor <= n.div_euclid(divisor) {
        if n.is_multiple_of(divisor) {
            return false;
        }
        divisor = divisor
            .checked_add(2)
            .expect("bounded trial-divisor fixture");
    }
    true
}
