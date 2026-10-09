//! Sliding-window powers against binary and exact-power oracles.

use proptest::prelude::*;

use super::{
    BarrettDomain, InternalMpUint, LIMB_BITS, Limb, MontgomeryDomain, MontgomeryScratch, MulScratch,
};

#[test]
fn small_windows_and_unit_moduli_match_exact_powers() {
    for width in [1, 4, 5] {
        let odd = InternalMpUint::one()
            .shl(width * LIMB_BITS)
            .add(&InternalMpUint::from_limb(3));
        for modulus in [
            InternalMpUint::one(),
            InternalMpUint::from_limb(2),
            odd.clone(),
            odd.add(&InternalMpUint::one()),
        ] {
            let barrett = BarrettDomain::new(&modulus);
            let mut multiplication = MulScratch::default();
            for base in [
                InternalMpUint::zero(),
                InternalMpUint::one(),
                modulus.add(&InternalMpUint::from_limb(7)),
            ] {
                for exponent in 0..=7 {
                    let expected = base.pow(exponent).rem(&modulus);
                    let exp = InternalMpUint::from_u64(u64::from(exponent));
                    assert_eq!(base.pow_mod(&exp, &modulus), expected);
                    assert_eq!(barrett.pow(&base, &exp, &mut multiplication), expected);
                    if modulus.is_odd() {
                        let montgomery = MontgomeryDomain::new::<true>(&modulus);
                        let mut scratch = MontgomeryScratch::default();
                        assert_eq!(montgomery.pow(&base, &exp, &mut scratch, false), expected);
                        let mut raw = montgomery.pow(&base, &exp, &mut scratch, true);
                        let mut decoded = InternalMpUint::zero();
                        montgomery.reduce_into(&mut raw, &mut decoded, &mut scratch);
                        assert_eq!(decoded, expected);
                    }
                }
            }
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Long exponent ladders run natively; the small-window and binary-oracle properties cover scalar, inline, and general powers under Miri."
)]
fn truncated_power_tables_match_binary_at_native_and_window_boundaries() {
    for width in 1..=5 {
        let modulus = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let base = modulus.sub(&InternalMpUint::from_limb(7));
        let montgomery = MontgomeryDomain::new::<true>(&modulus);
        let barrett = BarrettDomain::new(&modulus);
        let mut scratch = MontgomeryScratch::default();
        let mut multiplication = MulScratch::default();
        for bits in [
            7,
            LIMB_BITS - 1,
            LIMB_BITS,
            LIMB_BITS + 1,
            63,
            64,
            65,
            255,
            256,
            257,
            1023,
            1024,
            1025,
        ] {
            let high = InternalMpUint::power_of_two(bits - 1);
            let mut alternating = InternalMpUint::zero();
            for position in (0..bits).rev().step_by(2) {
                alternating.add_assign(&InternalMpUint::power_of_two(position));
            }
            for exponent in [
                high.add(&InternalMpUint::from_limb(7)),
                high.add(&InternalMpUint::from_limb(63)),
                high.shl(1).sub(&InternalMpUint::one()),
                alternating,
            ] {
                let expected = binary_reference(&base, &exponent, &modulus);
                assert_eq!(base.pow_mod(&exponent, &modulus), expected);
                assert_eq!(
                    montgomery.pow(&base, &exponent, &mut scratch, false),
                    expected
                );
                assert_eq!(barrett.pow(&base, &exponent, &mut multiplication), expected);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 512 }))]
    #[test]
    fn scalar_and_inline_domains_match_binary_reduction(
        mut words in prop_oneof![proptest::collection::vec(any::<Limb>(), 1..=1), proptest::collection::vec(any::<Limb>(), 2..=4)],
        base_words in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 40 }),
        exponent_words in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 1 } else { 8 }),
    ) {
        *words.first_mut().expect("positive modulus width") |= 1;
        *words.last_mut().expect("positive modulus width") |= 1;
        let modulus = InternalMpUint::from_limbs(words);
        let base = InternalMpUint::from_limbs(base_words);
        let exponent = InternalMpUint::from_limbs(exponent_words);
        prop_assert_eq!(base.pow_mod(&exponent, &modulus), binary_reference(&base, &exponent, &modulus));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 256 }))]
    #[test]
    fn sparse_exponents_match_binary_reduction_in_both_domains(
        words in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 5 } else { 34 }),
        base_words in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 40 }),
        high_bit in 0_usize..=if cfg!(miri) { 65 } else { 256 },
        low_bit in 0_usize..=if cfg!(miri) { 65 } else { 256 },
        third_bit in proptest::option::of(0_usize..=if cfg!(miri) { 65 } else { 256 }),
    ) {
        let mut modulus = InternalMpUint::from_limbs(words);
        if modulus.is_even() { modulus.increment(); }
        let base = InternalMpUint::from_limbs(base_words);
        let mut exponent = InternalMpUint::one().shl(high_bit).add(&InternalMpUint::one().shl(low_bit));
        if let Some(bit) = third_bit { exponent.add_assign(&InternalMpUint::one().shl(bit)); }
        let expected = binary_reference(&base, &exponent, &modulus);
        prop_assert_eq!(&base.pow_mod(&exponent, &modulus), &expected);
        prop_assert_eq!(&MontgomeryDomain::new::<true>(&modulus).pow(&base, &exponent, &mut MontgomeryScratch::default(), false), &expected);
        prop_assert_eq!(&BarrettDomain::new(&modulus).pow(&base, &exponent, &mut MulScratch::default()), &expected);
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "64-to-128-limb domain comparisons require native execution; small-window and independent binary-power properties run under Miri."
)]
fn domains_agree_across_recursive_reduction_widths() {
    let repeated_nibble = Limb::MAX.div_euclid(15);
    for width in [64, 96, 128] {
        let mut base_words = alloc::vec![repeated_nibble.saturating_mul(10); width];
        *base_words.first_mut().expect("nonempty base") |= 1;
        let base = InternalMpUint::from_limbs(base_words);
        let mut exponent_words = alloc::vec![repeated_nibble.saturating_mul(5); 4];
        *exponent_words.first_mut().expect("nonempty exponent") |= 1;
        let exponent = InternalMpUint::from_limbs(exponent_words);
        let mut modulus_words = alloc::vec![repeated_nibble.saturating_mul(14); width];
        *modulus_words.first_mut().expect("nonempty modulus") |= 1;
        let modulus = InternalMpUint::from_limbs(modulus_words);
        let montgomery = MontgomeryDomain::new::<true>(&modulus).pow(
            &base,
            &exponent,
            &mut MontgomeryScratch::default(),
            false,
        );
        let barrett =
            BarrettDomain::new(&modulus).pow(&base, &exponent, &mut MulScratch::default());
        assert_eq!(&montgomery, &binary_reference(&base, &exponent, &modulus));
        assert_eq!(montgomery, barrett);
        assert_eq!(base.pow_mod(&exponent, &modulus), barrett);
    }
}

fn binary_reference(
    base: &InternalMpUint,
    exponent: &InternalMpUint,
    modulus: &InternalMpUint,
) -> InternalMpUint {
    let mut power = base.rem(modulus);
    let mut remaining = exponent.clone();
    let mut result = InternalMpUint::one().rem(modulus);
    while !remaining.is_zero() {
        if remaining.is_odd() {
            result = result.mul(&power).rem(modulus);
        }
        remaining.shr_assign(1);
        power = power.square().rem(modulus);
    }
    result
}
