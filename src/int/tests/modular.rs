//! Modular arithmetic properties.

use super::*;

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

        let actual = ArbiUint::from(base)
            .pow_mod(&ArbiUint::from(exponent), &ArbiUint::from(modulus))
            .expect("non-zero modulus");
        prop_assert_eq!(actual, ArbiUint::from(expected));
    }
}

proptest! {
    #[test]
    fn prop_pow_mod_barrett_fuzz(
        base in strategies::uint(4),
        exponent in strategies::uint(2),
        modulus_seed in strategies::uint(4),
    ) {
        let modulus = if modulus_seed.is_odd() {
            &modulus_seed - &ArbiUint::one()
        } else {
            modulus_seed
        };
        prop_assume!(!modulus.is_zero());
        let result = base
            .pow_mod(&exponent, &modulus)
            .expect("pow_mod returned None");
        prop_assert!(result < modulus);
    }
}

proptest! {
    #[test]
    fn prop_mod_add_commutative(m in strategies::uint_nonzero(8), a in strategies::uint(8), b in strategies::uint(8)) {
        if m.value.is_one() { return Ok(()); }
        let ab = ArbiUint::add_mod(&a, &b, &m);
        let ba = ArbiUint::add_mod(&b, &a, &m);
        prop_assert_eq!(ab, ba, "add_mod not commutative");
    }
}

proptest! {
    #[test]
    fn prop_mod_mul_commutative(m in strategies::uint_nonzero(6), a in strategies::uint(6), b in strategies::uint(6)) {
        if m.value.is_one() { return Ok(()); }
        let ab = ArbiUint::mul_mod(&a, &b, &m);
        let ba = ArbiUint::mul_mod(&b, &a, &m);
        prop_assert_eq!(ab, ba, "mul_mod not commutative");
    }
}

proptest! {
    #[test]
    fn prop_modular_inverse(m in strategies::uint_nonzero(4), a in strategies::uint(4)) {
        if m.value.is_zero() || m.value.is_one() { return Ok(()); }
        if a.value.is_zero() || a.value.gcd(&m.value).is_one() { return Ok(()); }
        if let Some(inv) = ArbiUint::invert(&a, &m) {
            let product = ArbiUint::mul_mod(&a, &inv, &m).expect("valid modulus");
            prop_assert_eq!(product, ArbiUint::one(), "a * invert(a, m) != 1 (mod m)");
        }
    }
}
