//! Exact-power boundaries, cached powers, and degrees beyond native precision.

use proptest::{prelude::ProptestConfig, prop_assert, proptest};

use super::super::{DoubleLimb, InternalMpUint, Limb, NthRootScratch, Roots};

#[test]
#[cfg_attr(
    miri,
    ignore = "Exhaustive native powers and large precision matrices require native execution; bounded cached-root and bracket properties run under Miri."
)]
fn exact_powers_bracket_native_and_precision_growth_boundaries() {
    for degree in 3..DoubleLimb::BITS {
        for digit in 2_u8..=127 {
            if let Some(power) = DoubleLimb::from(digit).checked_pow(degree) {
                let mask = DoubleLimb::try_from(Limb::MAX).expect("limb mask");
                let low = Limb::try_from(power & mask).expect("low half");
                let high = Limb::try_from(power >> Limb::BITS).expect("high half");
                check_power_boundary(
                    &InternalMpUint::from_limbs_2(low, high),
                    &InternalMpUint::from_limb(Limb::from(digit)),
                    degree,
                );
            }
        }
    }
    let one = InternalMpUint::one();
    for width in [1, 2, 3, 4, 5, 8, 17] {
        for degree in [3, 4, 5, 7, 16, 31] {
            for root in [
                one.shl(width * usize::try_from(Limb::BITS).expect("limb bits")),
                InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]),
            ] {
                check_power_boundary(&root.pow(degree), &root, degree);
            }
        }
    }
    for bits in [6, 7, 8, 16, 31, 32, 63, 64, 127, 128, 191, 192, 193, 257] {
        for degree in [3, 5, 7, 17, 31] {
            for digit in 0..=3 {
                let root = one.shl(bits).add(&InternalMpUint::from_limb(digit));
                check_power_boundary(&root.pow(degree), &root, degree);
            }
        }
    }
}

fn check_power_boundary(power: &InternalMpUint, root: &InternalMpUint, degree: u32) {
    let one = InternalMpUint::one();
    for (input, expected) in [
        (power.sub(&one), root.sub(&one)),
        (power.clone(), root.clone()),
        (power.add(&one), root.clone()),
    ] {
        assert_eq!(input.nth_root(degree), expected);
        if input.limbs().len() >= 2 {
            let mut scratch = NthRootScratch::default();
            let cached =
                scratch.nth_root_multi_limb::<true>(&input, degree, input.significant_bits());
            assert_eq!(cached, expected);
            assert_eq!(
                scratch.x_pow_n_minus_1,
                cached.pow(degree.checked_sub(1).expect("degree >= 2"))
            );
            assert_eq!(scratch.temp_prod, cached.pow(degree));
        }
    }
}

#[test]
fn native_overflow_and_large_degrees_preserve_exact_brackets() {
    let one = InternalMpUint::one();
    let value = InternalMpUint::from_u64(u64::MAX);
    assert_eq!(value.nth_root(33), InternalMpUint::from_limb(3));
    let wide_degree = Limb::BITS
        .checked_mul(2)
        .and_then(|bits| bits.checked_sub(1))
        .expect("double-limb degree");
    for degree in [3, 5, 7, 17, 31, wide_degree] {
        for high in [1, 2, Limb::MAX] {
            for low in [0, 1, Limb::MAX] {
                let input = InternalMpUint::from_limbs_2(low, high);
                let root = input.nth_root(degree);
                assert!(root.pow(degree) <= input);
                assert!(root.add(&one).pow(degree) > input);
                let mut scratch = NthRootScratch::default();
                assert_eq!(
                    scratch.nth_root_multi_limb::<true>(&input, degree, input.significant_bits()),
                    root
                );
                assert_eq!(
                    scratch.x_pow_n_minus_1,
                    root.pow(degree.checked_sub(1).expect("positive exponent"))
                );
                assert_eq!(scratch.temp_prod, root.pow(degree));
            }
        }
    }
    for degree in 2..=80 {
        let native = InternalMpUint::from_limb(Limb::MAX);
        let root = Roots::nth_root_single_limb(&native, degree);
        assert!(root.pow(degree) <= native);
        assert!(root.add(&one).pow(degree) > native);
    }
    let wide = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 5]);
    let bits = u32::try_from(wide.significant_bits()).expect("bounded fixture");
    for degree in [
        bits,
        bits.checked_add(1).expect("degree successor"),
        u32::MAX,
    ] {
        assert_eq!(wide.nth_root(degree), one);
    }
    let two = InternalMpUint::from_limb(2);
    for degree in [31, 32, 33, 63, 64, 65, 127] {
        let power = two.pow(degree);
        assert_eq!(power.nth_root(degree), two);
        assert_eq!(power.sub(&one).nth_root(degree), one);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 64 }))]
    #[test]
    fn native_roots_bracket_every_admitted_degree(value in 1_usize..=Limb::MAX, degree in 2_u32..=80) {
        let input = InternalMpUint::from_limb(value);
        let root = Roots::nth_root_single_limb(&input, degree);
        prop_assert!(root.pow(degree) <= input);
        prop_assert!(root.add(&InternalMpUint::one()).pow(degree) > input);
    }
}
