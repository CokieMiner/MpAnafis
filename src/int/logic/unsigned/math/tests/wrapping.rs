//! Power-of-two residues, storage reuse, and bounded bit extraction.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "Test widths are bounded by 84 limbs; bit offsets, capacities, and range endpoints stay below 6000 on every supported target."
)]

use proptest::{
    prelude::{ProptestConfig, any},
    prop_assert_eq, proptest,
};

use super::super::{InternalMpUint, LIMB_BITS, Limb};

#[test]
fn wrapping_preserves_heap_capacity_for_zero_and_truncated_residues() {
    for width in [
        0,
        1,
        LIMB_BITS - 1,
        LIMB_BITS,
        4 * LIMB_BITS,
        5 * LIMB_BITS + 1,
    ] {
        let modulus = InternalMpUint::power_of_two(width);
        for value in [
            InternalMpUint::zero(),
            InternalMpUint::one(),
            modulus.clone(),
            modulus.add(&InternalMpUint::one()),
            InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 8]),
        ] {
            let residue = value.rem(&modulus);
            let negative = if residue.is_zero() {
                InternalMpUint::zero()
            } else {
                modulus.sub(&residue)
            };
            let mut wrapped = InternalMpUint::with_capacity(16);
            wrapped.clone_from(&value);
            let capacity = wrapped.capacity();
            wrapped = wrapped.apply_wrapping(width);
            assert_eq!(wrapped, residue);
            assert_eq!(wrapped.capacity(), capacity);
            let mut negated = InternalMpUint::with_capacity(16);
            negated.clone_from(&value);
            let negative_capacity = negated.capacity();
            negated = negated.apply_negate_wrapping(width);
            assert_eq!(negated, negative);
            assert_eq!(negated.capacity(), negative_capacity);
        }
    }
    // An arbitrary width cannot force allocation when the retained residue is zero.
    assert!(
        InternalMpUint::zero()
            .apply_negate_wrapping(usize::MAX)
            .is_zero()
    );
}

#[test]
fn wrapping_trims_zero_prefixes_after_partial_and_whole_limb_reduction() {
    for width in [1, 4, 5, 16] {
        for retained in 0..width {
            let mut limbs = alloc::vec![0; width];
            *limbs.last_mut().expect("nonempty test magnitude") = 1;
            if retained != 0 {
                *limbs.first_mut().expect("nonempty test magnitude") = 7;
            }
            let value = InternalMpUint::from_limbs(limbs);
            for bits in [retained * LIMB_BITS, retained * LIMB_BITS + 1] {
                let expected = value.rem(&InternalMpUint::power_of_two(bits));
                let mut destination = InternalMpUint::with_capacity(32);
                destination.clone_from(&value);
                let allocation = destination.limbs().as_ptr();
                let capacity = destination.capacity();
                destination = destination.apply_wrapping(bits);
                assert_eq!(destination, expected);
                assert_eq!(destination.capacity(), capacity);
                assert_eq!(destination.limbs().as_ptr(), allocation);
            }
        }
    }
    let value = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 5]);
    assert_eq!(value.clone().apply_wrapping(usize::MAX), value);
}

#[test]
fn negation_trims_all_ones_suffixes_to_the_first_nonzero_column() {
    for width in [1, 4, 5, 16] {
        for first in 0..width {
            let mut limbs = alloc::vec![Limb::MAX; width];
            for limb in limbs.iter_mut().take(first) {
                *limb = 0;
            }
            let value = InternalMpUint::from_limbs(limbs);
            for bits in [width * LIMB_BITS, width * LIMB_BITS - 1] {
                let modulus = InternalMpUint::power_of_two(bits);
                let expected = modulus.sub(&value.rem(&modulus));
                let mut destination = InternalMpUint::with_capacity(32);
                destination.clone_from(&value);
                let allocation = destination.limbs().as_ptr();
                let capacity = destination.capacity();
                destination = destination.apply_negate_wrapping(bits);
                assert_eq!(destination, expected);
                assert_eq!(destination.limbs().len(), first + 1);
                assert_eq!(destination.capacity(), capacity);
                assert_eq!(destination.limbs().as_ptr(), allocation);
            }
        }
    }
}

#[test]
fn negation_clears_partial_top_only_residues_and_overwrites_inactive_storage() {
    for width in [1, 4, 5, 16] {
        let mut limbs = alloc::vec![0; width];
        *limbs.last_mut().expect("nonempty test magnitude") = 1 << (LIMB_BITS - 1);
        let value = InternalMpUint::from_limbs(limbs);
        for bits in [0, (width - 1) * LIMB_BITS, width * LIMB_BITS - 1] {
            assert!(value.clone().apply_negate_wrapping(bits).is_zero());
        }

        // Reduction leaves nonzero inactive limbs in both representations.
        // Negation must initialize the whole extension independently of them.
        let original = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let narrowed = original.apply_wrapping(LIMB_BITS);
        let capacity = narrowed.capacity();
        for bits in [width * LIMB_BITS, width * LIMB_BITS + 1] {
            let modulus = InternalMpUint::power_of_two(bits);
            let expected = modulus.sub(&narrowed);
            let output = narrowed.clone().apply_negate_wrapping(bits);
            assert_eq!(output, expected);
        }
        let result = narrowed.apply_negate_wrapping(width * LIMB_BITS);
        assert_eq!(result.capacity(), capacity);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 32 }))]
    #[test]
    fn prop_wrapping_and_negation_match_primitive_modular_arithmetic(
        input in any::<u128>(),
        bits in 0_usize..=128,
    ) {
        let mask = if bits == 128 { u128::MAX } else { (1_u128 << bits) - 1 };
        let residue = input & mask;
        let negative = input.wrapping_neg() & mask;
        let value = InternalMpUint::from_u128(input);
        let (wrapped, overflow) = value.clone().apply_wrapping_with_overflow(bits);
        prop_assert_eq!(wrapped, InternalMpUint::from_u128(residue));
        prop_assert_eq!(overflow, input != residue);
        prop_assert_eq!(value.apply_negate_wrapping(bits), InternalMpUint::from_u128(negative));
    }

    #[test]
    fn prop_wrapping_and_negation_match_power_of_two_remainders(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=80),
        bits in 0_usize..=84 * LIMB_BITS,
    ) {
        let value = InternalMpUint::from_limbs(limbs);
        let modulus = InternalMpUint::power_of_two(bits);
        let residue = value.rem(&modulus);
        let negative = if residue.is_zero() { InternalMpUint::zero() } else { modulus.sub(&residue) };
        let (wrapped, overflow) = value.clone().apply_wrapping_with_overflow(bits);
        prop_assert_eq!(overflow, value != residue);
        prop_assert_eq!(wrapped, residue);
        prop_assert_eq!(value.apply_negate_wrapping(bits), negative);
    }

}
