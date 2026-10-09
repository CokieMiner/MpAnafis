//! Shifted coefficient accumulation against ordinary integer arithmetic.

#![expect(
    unsafe_code,
    reason = "Test allocations contain complete coefficients and biased accumulators with explicit carry space"
)]

use crate::int::logic::unsigned::InternalMpUint;

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 48 }))]

    #[test]
    fn shifted_accumulation_matches_integer_addition_and_subtraction(
        words in prop::collection::vec(any::<Limb>(), 1..=8),
        shift_limbs in 0_usize..4,
        bit_shift in 0_u32..Limb::BITS,
        tail in prop::collection::vec(any::<Limb>(), 0..=4),
        cut in 0_usize..8,
    ) {
        let width = words.len();
        let total = shift_limbs.checked_add(width).and_then(|n| n.checked_add(tail.len())).and_then(|n| n.checked_add(2)).expect("test accumulator fits");
        let shift = shift_limbs.checked_mul(LIMB_BITS).and_then(|n| n.checked_add(usize::try_from(bit_shift).expect("shift fits"))).expect("test offset fits");
        let mut base = vec![0; total];
        base.get_mut(..tail.len()).expect("tail fits").copy_from_slice(&tail);
        let magnitude = InternalMpUint::from_limbs_slice(&words).shl(shift);
        let expected_sum = InternalMpUint::from_limbs_slice(&base).add(&magnitude);
        let mut actual_sum = base;
        // SAFETY: words is nonempty and both disjoint spans contain the
        // shifted coefficient plus two zero high limbs, so its sum fits.
        unsafe {
            SsaCoefficients::process_positive_coeff(
                &words, NonZeroUsize::new(width).expect("positive width"), shift_limbs, bit_shift, &mut actual_sum,
            );
        }
        prop_assert_eq!(InternalMpUint::from_limbs_slice(&actual_sum), expected_sum);

        let bound = width.saturating_sub(cut).max(1);
        for guard_only in [false, true] {
            let mut coefficient = vec![Limb::MAX; width.checked_add(1).expect("guard fits")];
            let mut negative_magnitude = vec![0; bound];
            if guard_only {
                coefficient.fill(0);
                *coefficient.last_mut().expect("guard") = 1;
                *negative_magnitude.first_mut().expect("positive bound") = 1;
            } else {
                coefficient.get_mut(..bound).expect("bound fits").copy_from_slice(words.get(..bound).expect("bound fits"));
                *coefficient.first_mut().expect("nonempty data") |= 3;
                *coefficient.get_mut(width.checked_sub(1).expect("positive width")).expect("top data") |= 1 << (Limb::BITS - 1);
                *coefficient.last_mut().expect("guard") = 0;
                let mut carry = 2;
                for (digit, &source) in negative_magnitude.iter_mut().zip(&coefficient) {
                    let (value, escaped) = (!source).overflowing_add(carry);
                    *digit = value;
                    carry = Limb::from(escaped);
                }
                prop_assert_eq!(carry, 0);
            }
            let mut actual = vec![Limb::MAX; total];
            let expected = InternalMpUint::from_limbs_slice(&actual).sub(&InternalMpUint::from_limbs(negative_magnitude).shl(shift));
            // SAFETY: the canonical negative magnitude is strictly below
            // B^bound; the all-ones accumulator covers its shift and high bias.
            unsafe {
                SsaCoefficients::shift_sub_magnitude_run(
                    &mut actual, shift_limbs, &coefficient, NonZeroUsize::new(bound).expect("positive bound"), width, bit_shift,
                );
            }
            prop_assert_eq!(InternalMpUint::from_limbs(actual), expected);
        }
    }
}

#[test]
fn aligned_negative_magnitudes_absorb_complete_borrow_chains() {
    for bound in [1_usize, 2, 4, 8] {
        let inner = bound + 1;
        let mut sparse = vec![0; bound];
        *sparse.last_mut().expect("positive bound") = 1;
        let mut magnitudes = vec![vec![Limb::MAX; bound], sparse];
        for small in [1, 2] {
            let mut magnitude = vec![0; bound];
            *magnitude.first_mut().expect("positive bound") = small;
            magnitudes.push(magnitude);
        }
        for magnitude in magnitudes {
            let mut coefficient = vec![0; inner + 1];
            coefficient
                .get_mut(..bound)
                .expect("bound fits")
                .copy_from_slice(&magnitude);
            // SAFETY: magnitude is canonical and below B^bound; the complete
            // coefficient includes inner data limbs and its zero guard.
            unsafe {
                SsaRing::negate(&mut coefficient, inner * LIMB_BITS);
            }
            for shift in [0, 3] {
                let mut actual = vec![0; shift + bound + 4];
                *actual.last_mut().expect("high bias") = 1;
                let expected = InternalMpUint::from_limbs_slice(&actual)
                    .sub(&InternalMpUint::from_limbs_slice(&magnitude).shl(shift * LIMB_BITS));
                // SAFETY: the disjoint complete negative coefficient has a
                // strict bound, and the high bias exceeds its shifted magnitude.
                unsafe {
                    SsaCoefficients::shift_sub_magnitude_run(
                        &mut actual,
                        shift,
                        &coefficient,
                        NonZeroUsize::new(bound).expect("positive bound"),
                        inner,
                        0,
                    );
                }
                assert_eq!(InternalMpUint::from_limbs(actual), expected);
            }
        }
    }
    for low in [0, 1, Limb::MAX] {
        let mut actual = [low, 1];
        let mut expected = actual;
        // SAFETY: each array contains one initialized data limb and its guard;
        // the two-limb accumulator has no digits beyond twice the data width.
        unsafe {
            let _ = SsaRing::normalize(&mut expected, LIMB_BITS);
            SsaCoefficients::fold_high_into_low(&mut actual, 1);
        }
        assert_eq!(actual, expected);
    }
}
