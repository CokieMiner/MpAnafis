//! Semi-normal guard combinations and addition carry boundaries.

#![expect(
    unsafe_code,
    clippy::indexing_slicing,
    reason = "Bounded test coefficients include the guard limb and remain disjoint throughout modular addition"
)]

use alloc::vec;

use super::{super::super::tests::oracle_add_mod, LIMB_BITS, Limb, SsaRing};

#[test]
fn guard_combinations_and_carries_match_independent_modular_addition() {
    for limbs in [1, 3, 4, 127, 128, 129] {
        if cfg!(miri) && limbs > 4 {
            continue;
        }
        let bits = limbs * LIMB_BITS;
        for left_fill in [0, 1, Limb::MAX] {
            for right_fill in [0, 1, Limb::MAX] {
                for (left_guard, right_guard) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let mut left = vec![left_fill; limbs + 1];
                    let mut right = vec![right_fill; limbs + 1];
                    left[limbs] = left_guard;
                    right[limbs] = right_guard;
                    let mut canonical_left = left.clone();
                    let mut canonical_right = right.clone();
                    // SAFETY: complete semi-normalized slots are canonicalized
                    // before entering the oracle's stricter residue domain.
                    unsafe {
                        let _ = SsaRing::normalize(&mut canonical_left, bits);
                        let _ = SsaRing::normalize(&mut canonical_right, bits);
                    }
                    let expected = oracle_add_mod(&canonical_left, &canonical_right, bits);
                    // SAFETY: complete disjoint semi-normalized ring coefficients.
                    unsafe {
                        SsaRing::add_in_place(&mut left, &right, bits);
                        assert!(left[limbs] <= 1);
                        let _ = SsaRing::normalize(&mut left, bits);
                    }
                    assert_eq!(
                        left, expected,
                        "limbs={limbs}, guards={left_guard},{right_guard}"
                    );
                }
            }
        }
    }
}
