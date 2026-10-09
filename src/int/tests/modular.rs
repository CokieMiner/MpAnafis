//! Modular arithmetic properties.

use proptest::prelude::{any, prop_assert, prop_assert_eq, proptest};

use crate::{MpUint, Precision};

use super::{strategies, support::nz};

proptest! {
    #[test]
    fn prop_pow_mod_matches_primitive(
        base in any::<u64>(),
        exponent in any::<u32>(),
        modulus in 1_u64..=u64::MAX,
    ) {
        let primitive_modulus = u128::from(modulus);
        let mut expected = 1_u128 % primitive_modulus;
        let mut factor = u128::from(base) % primitive_modulus;
        let mut remaining_exponent = exponent;
        while remaining_exponent != 0 {
            if remaining_exponent & 1 == 1 {
                expected = (expected * factor) % primitive_modulus;
            }
            factor = (factor * factor) % primitive_modulus;
            remaining_exponent >>= 1;
        }

        let actual = MpUint::from(base)
            .pow_mod(&MpUint::from(exponent), &MpUint::from(modulus))
            .expect("non-zero modulus");
        prop_assert_eq!(actual, MpUint::from(expected));
    }
}

proptest! {
    #[test]
    fn even_modular_powers_obey_residue_and_successor_identities(
        base in strategies::uint(4),
        exponent in strategies::uint(2),
        modulus_seed in strategies::uint(4),
    ) {
        let modulus = (modulus_seed + MpUint::one()) << 1_usize;
        let result = base
            .pow_mod(&exponent, &modulus)
            .expect("pow_mod returned None");
        prop_assert!(result < modulus);
        prop_assert_eq!(&result, &(&base % &modulus).pow_mod(&exponent, &modulus).expect("positive modulus"));
        let successor = base.pow_mod(&(&exponent + MpUint::one()), &modulus).expect("positive modulus");
        prop_assert_eq!(successor, &result * &base % &modulus);
    }
}

proptest! {
    #[test]
    fn modular_arithmetic_matches_remainders_and_inverse_identities(
        modulus_seed in strategies::uint(16), left in strategies::uint(16), right in strategies::uint(16),
        bounded in any::<bool>(),
    ) {
        let precision = if bounded { Precision::Bounded(nz(1024)) } else { Precision::Unlimited };
        let a = if bounded { MpUint::with_precision_checked(left.clone(), nz(1024)).expect("sixteen words fit") } else { left.clone() };
        let b = if bounded { MpUint::with_precision_checked(right.clone(), nz(1024)).expect("sixteen words fit") } else { right.clone() };
        for seed in [modulus_seed, MpUint::one(), MpUint::zero()] {
            let modulus = if bounded { MpUint::with_precision_checked(seed.clone(), nz(1024)).expect("sixteen words fit") } else { seed.clone() };
            if seed.is_zero() {
                prop_assert_eq!(a.add_mod(&b, &modulus), None);
                prop_assert_eq!(a.sub_mod(&b, &modulus), None);
                prop_assert_eq!(a.mul_mod(&b, &modulus), None);
                prop_assert_eq!(a.invert(&modulus), None);
                prop_assert_eq!(a.pow_mod(&b, &modulus), None);
                continue;
            }
            let left_residue = &left % &seed;
            let right_residue = &right % &seed;
            let difference = if left_residue >= right_residue { &left_residue - &right_residue } else { &seed - &right_residue + &left_residue };
            for (actual, expected) in [
                (a.add_mod(&b, &modulus), (&left + &right) % &seed),
                (b.add_mod(&a, &modulus), (&left + &right) % &seed),
                (a.sub_mod(&b, &modulus), difference),
                (a.mul_mod(&b, &modulus), (&left * &right) % &seed),
                (b.mul_mod(&a, &modulus), (&left * &right) % &seed),
            ] {
                let result = actual.expect("positive modulus");
                prop_assert_eq!(&result, &expected);
                prop_assert_eq!(result.precision(), precision);
            }
            let inverse_result = a.invert(&modulus);
            prop_assert_eq!(inverse_result.is_some(), left.gcd(&seed).is_one());
            if let Some(inverse) = inverse_result {
                prop_assert_eq!(&left * (MpUint::zero() + &inverse) % &seed, MpUint::one() % &seed);
                prop_assert_eq!(inverse.precision(), precision);
            }
            if !seed.is_one() {
                let one = MpUint::one();
                let predecessor = &seed - &one;
                prop_assert_eq!(one.invert(&seed), Some(one));
                prop_assert_eq!(predecessor.invert(&seed), Some(predecessor));
            }
        }
    }
}
