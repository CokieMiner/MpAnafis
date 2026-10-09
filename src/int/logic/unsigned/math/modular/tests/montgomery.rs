//! Radix inverses and product reduction across scalar and recursive widths.

use alloc::vec;

use proptest::prelude::*;

use crate::int::{
    LIMB_BITS, Limb,
    logic::unsigned::{
        InternalMpUint,
        math::{Division, MONTGOMERY_CIOS_MAX_LIMBS, MontgomeryDomain, MontgomeryScratch},
    },
};

#[test]
#[cfg_attr(
    miri,
    ignore = "Radix inverses through 4096 limbs require native execution; bounded inverse properties exercise the product tier under Miri."
)]
fn radix_inverse_preserves_exact_congruence_across_product_tiers() {
    for width in [
        2_usize, 3, 4, 8, 16, 32, 33, 34, 63, 64, 65, 71, 72, 73, 127, 128, 129, 257, 512, 1_024,
        3_073, 4_096,
    ]
    .into_iter()
    .filter(|width| width.checked_mul(LIMB_BITS).is_some())
    {
        let bits = width.checked_mul(LIMB_BITS).expect("bounded radix width");
        let radix = InternalMpUint::one().shl(bits);
        for modulus in [
            radix.sub(&InternalMpUint::one()),
            radix.shr(1).add(&InternalMpUint::one()),
            radix.shr(LIMB_BITS).add(&InternalMpUint::one()),
        ] {
            let seed =
                Division::modular_inverse_limb(*modulus.limbs().first().expect("odd modulus"));
            let inverse = MontgomeryDomain::inverse_mod_radix(modulus.limbs(), seed);
            assert_eq!(inverse.len(), width);
            let value = InternalMpUint::from_limbs_slice(&inverse);
            assert_eq!(modulus.mul(&value).rem(&radix), InternalMpUint::one());
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Product storage matrices through 4096 limbs require native execution; bounded congruence properties run under Miri."
)]
fn reduction_preserves_radix_congruence_and_reuses_product_storage() {
    for width in [31_usize, 32, 33, 64, 65, 72, 73, 129, 257, 3_073, 4_096]
        .into_iter()
        .filter(|width| {
            width
                .checked_mul(LIMB_BITS)
                .and_then(|bits| bits.checked_mul(2))
                .is_some()
        })
    {
        let bits = width.checked_mul(LIMB_BITS).expect("bounded radix width");
        let capacity = width.checked_add(4).expect("bounded output capacity");
        let radix = InternalMpUint::one().shl(bits);
        for modulus in [
            radix.sub(&InternalMpUint::one()),
            radix.shr(1).add(&InternalMpUint::one()),
            radix.shr(LIMB_BITS).add(&InternalMpUint::one()),
        ] {
            let domain = MontgomeryDomain::new::<false>(&modulus);
            assert!(domain.r2.is_zero());
            let bound = modulus.shl(bits);
            let mut scratch = MontgomeryScratch::default();
            let mut output = InternalMpUint::with_capacity(capacity);
            let mut warm = bound.sub(&InternalMpUint::one());
            domain.reduce_into(&mut warm, &mut output, &mut scratch);
            let capacities = (
                scratch.coefficients.capacity(),
                scratch.high_product.capacity(),
                scratch.carry_product.capacity(),
            );
            for mut input in [
                bound.sub(&InternalMpUint::one()),
                bound.sub(&radix),
                modulus.sub(&InternalMpUint::one()).square(),
                modulus.clone(),
                InternalMpUint::one(),
                InternalMpUint::zero(),
            ] {
                let expected = input.rem(&modulus);
                domain.reduce_into(&mut input, &mut output, &mut scratch);
                assert!(output < modulus);
                assert_eq!(output.shl(bits).rem(&modulus), expected);
                assert_eq!(
                    (
                        scratch.coefficients.capacity(),
                        scratch.high_product.capacity(),
                        scratch.carry_product.capacity(),
                    ),
                    capacities,
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 64 }))]

    #[test]
    fn radix_inverse_matches_independent_congruence(
        mut modulus_limbs in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 34 } else { 2_049 }),
    ) {
        *modulus_limbs.first_mut().expect("nonempty modulus") |= 1;
        *modulus_limbs.last_mut().expect("nonempty modulus") |= 1;
        let modulus = InternalMpUint::from_limbs(modulus_limbs);
        let bits = modulus.limbs().len().checked_mul(LIMB_BITS).expect("bounded radix width");
        let radix = InternalMpUint::one().shl(bits);
        let domain = MontgomeryDomain::new::<false>(&modulus);
        prop_assert_eq!(domain.inverse.is_empty(), modulus.limbs().len() <= MONTGOMERY_CIOS_MAX_LIMBS);
        let seed = Division::modular_inverse_limb(*modulus.limbs().first().expect("odd modulus"));
        let coefficients = MontgomeryDomain::inverse_mod_radix(modulus.limbs(), seed);
        prop_assert_eq!(coefficients.len(), modulus.limbs().len());
        let inverse = InternalMpUint::from_limbs_slice(&coefficients);
        prop_assert_eq!(modulus.mul(&inverse).rem(&radix), InternalMpUint::one());
    }

    #[test]
    fn product_reduction_matches_independent_radix_congruence(
        mut modulus_limbs in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 5 } else { 513 }),
        input_limbs in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 11 } else { 1_027 }),
    ) {
        *modulus_limbs.first_mut().expect("nonempty modulus") |= 1;
        *modulus_limbs.last_mut().expect("nonempty modulus") |= 1;
        let modulus = InternalMpUint::from_limbs(modulus_limbs);
        let bits = modulus.limbs().len().checked_mul(LIMB_BITS).expect("bounded radix width");
        let bound = modulus.shl(bits);
        let mut input = InternalMpUint::from_limbs(input_limbs).rem(&bound);
        let expected = input.rem(&modulus);
        let domain = MontgomeryDomain::new::<false>(&modulus);
        let mut output = InternalMpUint::from_limbs(vec![Limb::MAX; if cfg!(miri) { 11 } else { 1_028 }]);
        domain.reduce_into(&mut input, &mut output, &mut MontgomeryScratch::default());
        prop_assert!(output < modulus);
        prop_assert_eq!(output.shl(bits).rem(&modulus), expected);
    }
}

#[cfg(not(target_pointer_width = "16"))]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    #[cfg_attr(miri, ignore = "Cyclic products at 3073-to-4096 limbs require native execution; scalar and bounded product reduction properties run under Miri.")]
    fn cyclic_reduction_matches_independent_radix_congruence(
        mut modulus_limbs in proptest::collection::vec(any::<Limb>(), 3_073..=4_096),
        input_limbs in proptest::collection::vec(any::<Limb>(), 6_146..=8_192),
    ) {
        *modulus_limbs.first_mut().expect("nonempty modulus") |= 1;
        *modulus_limbs.last_mut().expect("nonempty modulus") |= 1;
        let modulus = InternalMpUint::from_limbs(modulus_limbs);
        let bits = modulus.limbs().len().checked_mul(LIMB_BITS).expect("bounded radix width");
        let mut input = InternalMpUint::from_limbs(input_limbs).rem(&modulus.shl(bits));
        let expected = input.rem(&modulus);
        let domain = MontgomeryDomain::new::<false>(&modulus);
        let mut output = InternalMpUint::zero();
        domain.reduce_into(&mut input, &mut output, &mut MontgomeryScratch::default());
        prop_assert!(output < modulus);
        prop_assert_eq!(output.shl(bits).rem(&modulus), expected);
    }
}

#[test]
fn encoding_and_decoding_cross_reduction_tiers() {
    let widths = if cfg!(miri) {
        &[1, 4, 5][..]
    } else {
        &[1, 3, 4, 5, 31, 32, 33, 64][..]
    };
    for &width in widths {
        for top in [1, Limb::MAX] {
            for low in [1, 3, 5, Limb::MAX] {
                let mut limbs = vec![Limb::MAX; width];
                *limbs.last_mut().expect("nonempty modulus") = top;
                *limbs.first_mut().expect("nonempty modulus") = low;
                let modulus = InternalMpUint::from_limbs(limbs);
                let domain = MontgomeryDomain::new::<true>(&modulus);
                assert_eq!(low.wrapping_mul(domain.m_inv), Limb::MAX);
                let mut scratch = MontgomeryScratch::default();
                let mut product = InternalMpUint::with_capacity(2 * width + 2);
                let mut decoded = InternalMpUint::with_capacity(width + 2);
                for value in [
                    InternalMpUint::zero(),
                    InternalMpUint::one(),
                    modulus.sub(&InternalMpUint::one()),
                    modulus.clone(),
                    InternalMpUint::from_limbs(vec![Limb::MAX; width]),
                    InternalMpUint::from_limbs(vec![Limb::MAX; width + 1]),
                ] {
                    let mut encoded =
                        domain.transform_into_with_scratch(&value, &mut product, &mut scratch);
                    assert!(encoded < modulus);
                    domain.reduce_into(&mut encoded, &mut decoded, &mut scratch);
                    assert_eq!(decoded, value.rem(&modulus));
                }
            }
        }
    }
}

#[test]
fn reduction_carries_at_radix_boundaries() {
    let widths = if cfg!(miri) {
        &[1_usize, 4, 5][..]
    } else {
        &[1_usize, 3, 4, 5, 31, 32, 33, 64][..]
    };
    for &width in widths {
        let shift = width.checked_mul(LIMB_BITS).expect("small test radix");
        let radix = InternalMpUint::one().shl(shift);
        for modulus in [
            radix.sub(&InternalMpUint::one()),
            radix.shr(1).add(&InternalMpUint::one()),
            radix.shr(LIMB_BITS).add(&InternalMpUint::one()),
        ] {
            if modulus.is_even() {
                continue;
            }
            let domain = MontgomeryDomain::new::<true>(&modulus);
            let mut scratch = MontgomeryScratch::default();
            let bound = modulus.shl(shift);
            let mut output = InternalMpUint::zero();
            for mut input in [
                bound.sub(&InternalMpUint::one()),
                bound.sub(&modulus),
                modulus.sub(&InternalMpUint::one()).square(),
                InternalMpUint::one(),
                InternalMpUint::zero(),
            ] {
                let expected = input.rem(&modulus);
                domain.reduce_into(&mut input, &mut output, &mut scratch);
                assert!(output < modulus);
                assert_eq!(output.shl(shift).rem(&modulus), expected);
                if width <= 4 {
                    assert_eq!(output.capacity(), 4);
                }
            }
        }
    }
}
