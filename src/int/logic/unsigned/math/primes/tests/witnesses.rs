//! Base-two specialization and wide probable-prime screening.

use core::mem::swap;

use proptest::{
    prelude::{ProptestConfig, any},
    prop_assert, prop_assert_eq, proptest,
};

use super::super::{
    InternalMpUint, Limb, MontgomeryDomain, MontgomeryScratch, Primality, search::SIEVE_PRIMES,
};

#[test]
fn known_wide_primes_survive_each_fixed_base_policy() {
    for decimal in [
        "18446744073709551629",
        "340282366920938463463374607431768211507",
    ] {
        let prime = InternalMpUint::from_str_radix(decimal, 10).expect("decimal prime fixture");
        for &rounds in if cfg!(miri) {
            &[0, 1][..]
        } else {
            &[0, 1, 24, 64, u32::MAX][..]
        } {
            assert!(prime.is_probably_prime(rounds));
        }
        assert!(prime.is_prime());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 32 }))]
    #[test]
    fn base_two_round_matches_general_montgomery_powers(mut limbs in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 2 } else { 12 })) {
        *limbs.first_mut().expect("nonempty modulus") |= 1;
        *limbs.last_mut().expect("nonempty modulus") |= 1;
        let modulus = InternalMpUint::from_limbs(limbs);
        let domain = MontgomeryDomain::new::<true>(&modulus);
        let base = InternalMpUint::from_limb(2);
        let even = modulus.sub(&InternalMpUint::one());
        let twos = even.trailing_zeros();
        let odd = even.shr(twos);
        let mut scratch = MontgomeryScratch::default();
        let mut product = InternalMpUint::zero();
        let mut residue = InternalMpUint::zero();
        let one = domain.transform_into_with_scratch(&InternalMpUint::one(), &mut product, &mut scratch);
        let minus_one = domain.transform_into_with_scratch(&even, &mut product, &mut scratch);
        let mut value = domain.pow(&base, &odd, &mut scratch, true);
        let mut expected = value == one || value == minus_one;
        for _ in 1..twos {
            domain.square_into_with_scratch(&value, &mut residue, &mut product, &mut scratch);
            swap(&mut value, &mut residue);
            expected |= value == minus_one;
        }
        prop_assert_eq!(Primality::miller_rabin_test(&base, &odd, twos, &minus_one, &one, &mut product, &mut residue, &domain, &mut scratch), expected);
    }

    #[test]
    fn screened_small_prime_multiples_reject_wide_cofactors(
        words in proptest::collection::vec(any::<Limb>(), 3..=5),
        prime_index in 0_usize..SIEVE_PRIMES.len(),
    ) {
        let prime = *SIEVE_PRIMES.get(prime_index).expect("bounded prime index");
        let cofactor = InternalMpUint::from_limbs(words);
        let composite = cofactor.mul(&InternalMpUint::from_limb(prime));
        if composite.to_u64().is_none() {
            prop_assert!(!composite.is_probably_prime(24));
            prop_assert!(!composite.is_prime());
        }
    }
}
