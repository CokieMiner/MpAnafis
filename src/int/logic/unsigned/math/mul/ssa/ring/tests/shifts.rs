//! Fermat shifts with explicit guard corrections and excluded scratch sentinels.

#![expect(
    unsafe_code,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    reason = "Bounded test rings provide complete disjoint initialized coefficients and reduced shift exponents"
)]

use alloc::vec;

use proptest::prelude::*;

use super::{super::super::tests::oracle_shift, LIMB_BITS, Limb, SsaRing, oracle_negation};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 3 } else { 32 }))]

    #[test]
    fn both_shift_forms_match_independent_modular_doubling(
        data in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 4 } else { 129 }),
        guard in 0_usize..=1, exponent in any::<usize>(),
    ) {
        let bits = data.len() * LIMB_BITS;
        check_shift(&data, guard, exponent.rem_euclid(2 * bits));
    }
}

#[test]
fn complement_windows_cover_zero_prefixes_carries_and_half_periods() {
    for width in [1, 2, 3, 8] {
        let bits = width * LIMB_BITS;
        for guard in [0, 1] {
            for pattern in [0, 1, Limb::MAX - 1, Limb::MAX] {
                let data = vec![pattern; width];
                for shift in [
                    0,
                    1,
                    LIMB_BITS - 1,
                    LIMB_BITS + 1,
                    (bits >> 1) + 1,
                    bits - 1,
                ] {
                    check_shift(&data, guard, shift);
                }
            }
        }
    }
    for width in [1, 4, 8, 16, 33, 65, 129] {
        if cfg!(miri) && width > 4 {
            continue;
        }
        let bits = width * LIMB_BITS;
        for low in [0, 1, 2, 3, Limb::MAX] {
            for guard in [0, 1] {
                let mut sparse = vec![0; width];
                sparse[0] = low;
                check_shift(&sparse, guard, bits);
            }
        }
    }
}

fn check_shift(data: &[Limb], guard: Limb, shift: usize) {
    let width = data.len();
    let bits = width * LIMB_BITS;
    let negated = oracle_negation(data, guard);
    let canonical = oracle_negation(&negated[..width], negated[width]);
    let expected = oracle_shift(&canonical, shift, bits);
    let mut source = data.to_vec();
    source.push(guard);
    let mut actual = vec![37; width + 3];
    let mut separate = actual.clone();
    actual[1..=width + 1].copy_from_slice(&source);
    let mut scratch = vec![Limb::MAX; width + 3];
    // SAFETY: source and the exact destination and scratch windows are complete,
    // initialized, disjoint coefficients with guards<=1. shift<2*bits and
    // all sentinels remain outside the borrowed windows.
    unsafe {
        SsaRing::shift_in_place(
            &mut actual[1..=width + 1],
            shift,
            bits,
            &mut scratch[1..=width + 1],
        );
        SsaRing::shift_from(&mut separate[1..=width + 1], &source, shift, bits);
        let _ = SsaRing::normalize(&mut actual[1..=width + 1], bits);
        let _ = SsaRing::normalize(&mut separate[1..=width + 1], bits);
    }
    assert_eq!(&actual[1..=width + 1], expected);
    assert_eq!(&separate[1..=width + 1], expected);
    assert_eq!(
        (
            actual[0],
            actual[width + 2],
            separate[0],
            separate[width + 2]
        ),
        (37, 37, 37, 37)
    );
    assert_eq!((scratch[0], scratch[width + 2]), (Limb::MAX, Limb::MAX));
}
