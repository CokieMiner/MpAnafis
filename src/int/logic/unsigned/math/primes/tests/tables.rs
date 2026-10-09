//! Grouped prime products, inverse certificates, and wide screening residues.

use alloc::vec::Vec;

use proptest::{
    prelude::{ProptestConfig, any},
    prop_assert_eq, proptest,
};

use super::{
    super::{Gcd, InternalMpUint, Limb, Primality, TRIAL_PRIME_LIMIT, TRIAL_SCREEN},
    native::is_prime_usize,
};

#[test]
fn prime_products_and_certificates_cover_each_odd_prime_once() {
    let primes: Vec<_> = (3..TRIAL_PRIME_LIMIT)
        .step_by(2)
        .filter(|&p| is_prime_usize(p))
        .collect();
    assert_eq!(primes.len(), 302);
    let mut covered = Vec::new();
    let mut first = 0;
    for group in &TRIAL_SCREEN.products {
        let product = group.divisor;
        assert_eq!(product.wrapping_mul(group.inverse), 1);
        let mut remaining = product;
        let mut certificates = TRIAL_SCREEN
            .factors
            .get(first..group.end)
            .expect("constant endpoints")
            .iter();
        for &prime in &primes {
            if remaining.is_multiple_of(prime) {
                covered.push(prime);
                remaining = remaining.div_euclid(prime);
                let certificate = certificates.next().expect("prime has a certificate");
                assert_eq!(prime.wrapping_mul(certificate.inverse), 1);
                assert_eq!(certificate.limit, Limb::MAX.div_euclid(prime));
            }
        }
        assert_eq!(remaining, 1);
        assert!(certificates.next().is_none());
        if let Some(&next) = primes.get(group.end) {
            assert!(product.checked_mul(next).is_none(), "each group is maximal");
        }
        first = group.end;
    }
    assert_eq!(covered, primes);
    assert_eq!(first, TRIAL_SCREEN.factors.len());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]
    #[test]
    fn grouped_screen_matches_exact_remainders(mut limbs in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 3 } else { 24 })) {
        *limbs.first_mut().expect("nonempty input") |= 1;
        *limbs.last_mut().expect("nonempty input") |= 1;
        let input = InternalMpUint::from_limbs(limbs);
        let expected = TRIAL_SCREEN.products.iter().all(|group| {
            let product = group.divisor;
            let remainder = input.rem(&InternalMpUint::from_limb(product));
            let [low, _, _, _] = remainder.extract_4();
            Gcd::gcd_1(low, product) == 1
        });
        prop_assert_eq!(Primality::trial_division(&input), expected);
    }
}
