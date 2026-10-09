//! Triangular products, overflow repair, and untouched output contracts.

use proptest::prelude::*;

use crate::int::types::INLINE_LIMBS;

use super::{DivScratch, Division, InternalMpUint, Limb, PreparedDivisor};

#[test]
fn full_windows_reuse_leading_remainders_at_radix_boundaries() {
    for width in [3, 4, 5, 8] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        for digits in [1, width, width * 3] {
            let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; digits]);
            for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
                let numerator = divisor.mul(&expected).add(&residue);
                let mut scratch = DivScratch::default();
                let mut quotient = InternalMpUint::zero();
                let mut remainder = InternalMpUint::zero();
                let _ = Division::algorithm_d::<true, false, false, false>(
                    numerator.limbs(),
                    divisor.limbs(),
                    &mut quotient,
                    &mut remainder,
                    &mut scratch,
                );
                assert_eq!(quotient, expected);
                let _ = Division::algorithm_d::<false, true, false, false>(
                    numerator.limbs(),
                    divisor.limbs(),
                    &mut quotient,
                    &mut remainder,
                    &mut scratch,
                );
                assert_eq!(remainder, residue);
            }
        }
    }
}

proptest! {
    #[test]
    fn short_division_skips_zero_guards_with_remainder_certification(
        mut denominator in proptest::collection::vec(any::<Limb>(), 3..=80),
        mut digits in proptest::collection::vec(any::<Limb>(), 78),
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1 << (Limb::BITS - 1);
        let width = denominator.len();
        digits.truncate(width.checked_sub(2).expect("at least three divisor limbs"));
        *digits.last_mut().expect("positive quotient") |= 1;
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let prepared = PreparedDivisor::new(divisor.limbs());
        for residue in [InternalMpUint::zero(), InternalMpUint::one(), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&residue);
            let mut window = numerator.limbs().to_vec();
            let guard_len = width.checked_mul(2).and_then(|len| len.checked_sub(1)).expect("bounded guard width");
            window.resize(guard_len, 0);
            let mut output = alloc::vec![Limb::MAX; width.checked_sub(1).expect("nonempty quotient")];
            let certified = prepared.divide_quotient::<true, true, false>(
                &mut window, divisor.limbs(), &mut output,
            );
            prop_assert_eq!(InternalMpUint::from_limbs(output), expected.clone());
            if certified {
                prop_assert!(residue > expected);
            } else {
                prop_assert_eq!(InternalMpUint::from_limbs_slice(window.get(..width).expect("remainder span")), residue);
            }
        }
    }

    #[test]
    fn short_division_recovers_constructed_quotients(
        denominator in proptest::collection::vec(any::<Limb>(), 3..=80),
        digits in proptest::collection::vec(any::<Limb>(), 1..=80),
        shift in 0..Limb::BITS,
    ) {
        let mut den_limbs = denominator;
        let leading = den_limbs.last_mut().expect("three divisor limbs");
        *leading = (*leading >> shift) | (1 << Limb::BITS.wrapping_sub(shift).wrapping_sub(1));
        let divisor = InternalMpUint::from_limbs(den_limbs);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let mut quotient = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 100]);
        let mut unused = InternalMpUint::from_limb(17);
        let mut scratch = DivScratch::default();
        for remainder in [InternalMpUint::zero(), InternalMpUint::one(), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&remainder);
            if !Division::trivial::<true, false>(&numerator, &divisor, &mut quotient, &mut unused) {
                prop_assert!(!Division::algorithm_d::<true, false, false, false>(
                    numerator.limbs(), divisor.limbs(), &mut quotient, &mut unused, &mut scratch,
                ));
            }
            prop_assert_eq!(&quotient, &expected);
            prop_assert_eq!(&unused, &InternalMpUint::from_limb(17));
        }
    }
}

#[test]
fn short_division_clears_a_zero_quotient_without_touching_the_dividend() {
    for width in [3, 4, 5, 16, 40] {
        let divisor = alloc::vec![Limb::MAX; width];
        let mut dividend = alloc::vec![Limb::MAX >> 1; width];
        dividend.push(0);
        let original = dividend.clone();
        let mut quotient = [Limb::MAX];
        let prepared = PreparedDivisor::new(&divisor);
        assert!(!prepared.divide_quotient::<false, false, false>(
            &mut dividend,
            &divisor,
            &mut quotient,
        ));
        assert_eq!(quotient, [0]);
        assert_eq!(dividend, original);
    }
}

#[test]
fn short_division_handles_window_overflow_and_correction() {
    for width in [3, 4, 5, 16, 40] {
        let mut den = alloc::vec![0; width];
        *den.first_mut().expect("nonempty divisor") = Limb::MAX;
        *den.last_mut().expect("nonempty divisor") = 1 << Limb::BITS.wrapping_sub(1);
        let divisor = InternalMpUint::from_limbs(den);
        let expected = InternalMpUint::from_limb(Limb::MAX);
        // N = B*D-1 has quotient B-1, including a saturated high estimate.
        let numerator = divisor
            .mul(&expected)
            .add(&divisor.sub(&InternalMpUint::one()));
        let mut normalized = numerator.limbs().to_vec();
        normalized.push(0);
        let mut digits = alloc::vec![0; normalized.len() - width];
        let prepared = PreparedDivisor::new(divisor.limbs());
        assert!(!prepared.divide_quotient::<false, false, false>(
            &mut normalized,
            divisor.limbs(),
            &mut digits,
        ));
        assert_eq!(InternalMpUint::from_limbs(digits), expected);
        let mut quotient = InternalMpUint::from_limb(91);
        let mut untouched = InternalMpUint::from_limb(13);
        assert!(!Division::algorithm_d::<true, false, false, false>(
            numerator.limbs(),
            divisor.limbs(),
            &mut quotient,
            &mut untouched,
            &mut DivScratch::default(),
        ));
        assert_eq!(quotient, expected);
        assert_eq!(untouched, InternalMpUint::from_limb(13));
    }
}

#[test]
fn algorithm_d_keeps_fitting_quotients_inline_before_writing_a_zero_guard() {
    let mut scratch = DivScratch::default();
    for width in [INLINE_LIMBS - 1, INLINE_LIMBS, INLINE_LIMBS + 1] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let mut digits = alloc::vec![Limb::MAX; width];
        *digits.last_mut().expect("nonempty quotient") = 1;
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&residue);
            let mut quotient = InternalMpUint::zero();
            let mut remainder = InternalMpUint::zero();
            let _ = Division::algorithm_d::<true, true, false, false>(
                numerator.limbs(),
                divisor.limbs(),
                &mut quotient,
                &mut remainder,
                &mut scratch,
            );
            assert_eq!(quotient, expected);
            assert_eq!(remainder, residue);
            let quotient_only = numerator.div(&divisor);
            assert_eq!(quotient_only, expected);
            if width <= INLINE_LIMBS {
                assert_eq!(quotient.capacity(), INLINE_LIMBS);
                assert_eq!(quotient_only.capacity(), INLINE_LIMBS);
            }
        }
    }
}
