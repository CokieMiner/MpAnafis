//! Canonical residues, inverses, and retained subtraction storage.

use proptest::prelude::*;

use super::{InternalMpUint, LIMB_BITS, Limb};

#[test]
fn subtraction_complements_heap_residues_and_preserves_capacity() {
    for width in [3, 4, 5, 8] {
        let modulus = InternalMpUint::one().shl(width * LIMB_BITS);
        let mut output = InternalMpUint::with_capacity(width + 3);
        let capacity = output.capacity();
        for difference in [
            InternalMpUint::one(),
            modulus.sub(&InternalMpUint::one()),
            modulus.clone(),
            modulus.add(&InternalMpUint::one()),
            modulus.mul(&InternalMpUint::from_limb(3)),
            modulus
                .mul(&InternalMpUint::from_limb(3))
                .add(&InternalMpUint::one()),
        ] {
            let residue = difference.rem(&modulus);
            let expected = if residue.is_zero() {
                residue
            } else {
                modulus.sub(&residue)
            };
            InternalMpUint::zero().sub_mod_into(&difference, &modulus, &mut output);
            assert_eq!(output, expected);
            assert_eq!(output.capacity(), capacity);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]
    #[test]
    fn arithmetic_and_inverses_match_exact_residues(
        first in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 12 }),
        second in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 12 }),
        words in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 9 }),
    ) {
        let a = InternalMpUint::from_limbs(first);
        let b = InternalMpUint::from_limbs(second);
        let modulus = InternalMpUint::from_limbs(words).add(&InternalMpUint::one());
        let reduced_a = a.rem(&modulus);
        let reduced_b = b.rem(&modulus);
        let expected_difference = if reduced_a < reduced_b {
            modulus.sub(&reduced_b.sub(&reduced_a))
        } else {
            reduced_a.sub(&reduced_b)
        };
        let mut output = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 16]);
        a.sub_mod_into(&b, &modulus, &mut output);
        prop_assert_eq!(&output, &expected_difference);
        prop_assert_eq!(a.sub_mod(&b, &modulus), expected_difference);
        prop_assert!(a.sub_mod(&a, &modulus).is_zero());
        let sum = a.add_mod(&b, &modulus);
        prop_assert!(sum < modulus);
        prop_assert_eq!(&sum, &a.add(&b).rem(&modulus));
        prop_assert_eq!(sum, b.add_mod(&a, &modulus));
        let product = a.mul_mod(&b, &modulus);
        prop_assert!(product < modulus);
        prop_assert_eq!(&product, &a.mul(&b).rem(&modulus));
        prop_assert_eq!(product, b.mul_mod(&a, &modulus));
        let inverse = a.invert(&modulus);
        prop_assert_eq!(inverse.is_some(), a.gcd(&modulus).is_one());
        if let Some(value) = inverse {
            prop_assert!(value < modulus);
            prop_assert_eq!(a.mul(&value).rem(&modulus), InternalMpUint::one().rem(&modulus));
        }
    }
}
