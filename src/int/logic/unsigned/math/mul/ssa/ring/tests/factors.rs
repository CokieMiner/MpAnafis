//! Half-ring factors and square-root twists against independent modular shifts.

#![expect(
    unsafe_code,
    reason = "Complete disjoint test coefficients use one-bit guards and reduced exponents in positive limb-aligned rings"
)]

use alloc::vec;

use proptest::prelude::*;

use super::super::{
    super::tests::{oracle_add_mod, oracle_shift},
    LIMB_BITS, Limb, SsaRing,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 48 }))]

    #[test]
    fn half_ring_and_square_root_factors_match_independent_shifts(
        data in prop::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 4 } else { 8 }),
        guard in 0_usize..=1, shift in any::<usize>(),
    ) {
        let bits = data.len().checked_mul(LIMB_BITS).expect("test ring fits");
        let reduced = shift.rem_euclid(bits.checked_mul(2).expect("period fits"));
        let mut input = data;
        input.push(guard);
        let canonical = oracle_add_mod(&input, &vec![0; input.len()], bits);
        let rotated = oracle_shift(&canonical, bits >> 1, bits);
        let negative = oracle_shift(&canonical, bits, bits);
        let expected_half = oracle_add_mod(&rotated, &negative, bits);
        let positive_shift = (bits >> 2).checked_mul(3).and_then(|quarter| reduced.checked_add(quarter)).expect("test exponent fits");
        let negative_shift = reduced.checked_add(bits >> 2).and_then(|partial| partial.checked_add(bits)).expect("test exponent fits");
        let positive_sqrt = oracle_shift(&canonical, positive_shift, bits);
        let negative_sqrt = oracle_shift(&canonical, negative_shift, bits);
        let expected_sqrt = oracle_add_mod(&positive_sqrt, &negative_sqrt, bits);
        let mut actual_half = vec![Limb::MAX; input.len()];
        let mut actual_sqrt = input.clone();
        let mut separate_sqrt = vec![Limb::MAX; input.len()];
        let mut scratch = vec![Limb::MAX; input.len()];
        // SAFETY: initialized complete coefficients are pairwise disjoint,
        // every readable guard is at most one, and reduced<2*bits.
        unsafe {
            SsaRing::half_ring_sub_from(&mut actual_half, &input, bits);
            let _ = SsaRing::normalize(&mut actual_half, bits);
            SsaRing::shift_sqrt2(&mut actual_sqrt, reduced, bits, &mut scratch);
            let _ = SsaRing::normalize(&mut actual_sqrt, bits);
            SsaRing::shift_sqrt2_from(&mut separate_sqrt, &input, reduced, bits, &mut scratch);
            let _ = SsaRing::normalize(&mut separate_sqrt, bits);
        }
        prop_assert_eq!(actual_half, expected_half);
        prop_assert_eq!(&actual_sqrt, &expected_sqrt);
        prop_assert_eq!(separate_sqrt, expected_sqrt);
    }
}

#[test]
fn half_ring_factor_covers_guard_extremes_and_odd_limb_widths() {
    for width in [1_usize, 2, 3, 4, 63, 64, 65, 127, 128, 129, 256] {
        if cfg!(miri) && width > 4 {
            continue;
        }
        let bits = width.checked_mul(LIMB_BITS).expect("test ring fits");
        for fill in [0, 1, Limb::MAX] {
            for guard in [0, 1] {
                let mut input = vec![fill; width + 1];
                *input.last_mut().expect("guard") = guard;
                let mut actual = vec![Limb::MAX; input.len()];
                let mut expected = vec![0; input.len()];
                // SAFETY: the positive limb-aligned ring supplies complete
                // disjoint coefficients with guards<=1; bits/2<2*bits.
                unsafe {
                    SsaRing::shift_from(&mut expected, &input, bits >> 1, bits);
                    SsaRing::sub_in_place(&mut expected, &input, bits);
                    let _ = SsaRing::normalize(&mut expected, bits);
                    SsaRing::half_ring_sub_from(&mut actual, &input, bits);
                    let _ = SsaRing::normalize(&mut actual, bits);
                }
                assert_eq!(
                    actual, expected,
                    "width={width}, fill={fill}, guard={guard}"
                );
            }
        }
    }
}
