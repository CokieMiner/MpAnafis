//! Scalar reciprocals, overlapping outputs, and quotient-prefix guards.

use proptest::prelude::*;

use super::{DIVISION_STACK_LIMBS, DivScratch, Division, DoubleLimb, InternalMpUint, Limb};

#[test]
fn algorithm_d_normalization_boundary_covers_output_modes() {
    for top in [Limb::MAX, Limb::MAX >> 1] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![5, 7, top]);
        for width in [DIVISION_STACK_LIMBS - 1, DIVISION_STACK_LIMBS] {
            let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
            let mut scratch = DivScratch::default();
            let mut quotient = InternalMpUint::zero();
            let mut remainder = InternalMpUint::zero();
            let _ = Division::algorithm_d::<true, true, false, false>(
                numerator.limbs(),
                divisor.limbs(),
                &mut quotient,
                &mut remainder,
                &mut scratch,
            );
            let stack = width < DIVISION_STACK_LIMBS;
            assert_eq!(scratch.u_norm.len(), if stack { 0 } else { width + 1 });
            assert_eq!(
                scratch.v_norm.len(),
                if stack || top == Limb::MAX { 0 } else { 3 }
            );
            assert_eq!(quotient.mul(&divisor).add(&remainder), numerator);
            assert!(remainder < divisor);
            let mut q = InternalMpUint::from_limb(43);
            let mut r = InternalMpUint::from_limb(47);
            assert_eq!(
                Division::try_algorithm_d_unscratched::<true, false, false>(
                    &numerator, &divisor, &mut q, &mut r
                ),
                stack,
            );
            assert_eq!(r, InternalMpUint::from_limb(47));
            if stack {
                assert_eq!(q, quotient);
            } else {
                assert_eq!(q, InternalMpUint::from_limb(43));
            }
            let mut quotient_sentinel = InternalMpUint::from_limb(41);
            let mut remainder_only = InternalMpUint::zero();
            if stack {
                assert!(Division::try_algorithm_d_unscratched::<false, true, true>(
                    &numerator,
                    &divisor,
                    &mut quotient_sentinel,
                    &mut remainder_only
                ));
            } else {
                let _ = Division::algorithm_d::<false, true, false, false>(
                    numerator.limbs(),
                    divisor.limbs(),
                    &mut quotient_sentinel,
                    &mut remainder_only,
                    &mut scratch,
                );
            }
            assert_eq!(quotient_sentinel, InternalMpUint::from_limb(41));
            assert_eq!(remainder_only, remainder);
        }
    }
}

proptest! {
    #[test]
    fn scalar_division_matches_wide_digits_and_exact_overlap(
        seed in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
        limbs in proptest::collection::vec(any::<Limb>(), 0..=150),
    ) {
        let divisor = seed.max(1);
        let wide_divisor = DoubleLimb::try_from(divisor).expect("limb fits DoubleLimb");
        let mut expected_digits = alloc::vec![0; limbs.len()];
        let mut expected_remainder = 0;
        for (input, output) in limbs.iter().zip(expected_digits.iter_mut()).rev() {
            let dividend = (DoubleLimb::try_from(expected_remainder).expect("remainder fits")
                << Limb::BITS) | DoubleLimb::try_from(*input).expect("input fits");
            *output = Limb::try_from(dividend.div_euclid(wide_divisor)).expect("quotient fits");
            expected_remainder = Limb::try_from(dividend.rem_euclid(wide_divisor))
                .expect("remainder fits");
        }
        let expected = InternalMpUint::from_limbs(expected_digits);
        let mut quotient = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 160]);
        let remainder = Division::div_rem_1::<true>(&limbs, divisor, &mut quotient);
        prop_assert_eq!(&quotient, &expected);
        prop_assert_eq!(remainder, expected_remainder);
        let sentinel = quotient.clone();
        prop_assert_eq!(Division::div_rem_1::<false>(&limbs, divisor, &mut quotient),
            expected_remainder);
        prop_assert_eq!(quotient, sentinel);
        let mut in_place = InternalMpUint::from_limbs(limbs);
        prop_assert_eq!(Division::div_rem_1_assign(&mut in_place, divisor), expected_remainder);
        prop_assert_eq!(in_place, expected);
    }

    #[test]
    fn two_limb_streaming_division_preserves_values_and_unused_outputs(
        low in prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()],
        high in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
        mut digits in proptest::collection::vec(any::<Limb>(), 1..=160),
    ) {
        *digits.last_mut().expect("nonempty quotient") |= 1;
        let divisor = InternalMpUint::from_limbs_2(low, high.max(1));
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let mut quotient = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 170]);
        let mut remainder = quotient.clone();
        for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&residue);
            Division::div_rem_2_unnormalized::<true, true>(
                numerator.limbs(), low, high.max(1), &mut quotient, &mut remainder,
            );
            prop_assert_eq!((&quotient, &remainder), (&expected, &residue));
            let mut sentinel = InternalMpUint::from_limb(19);
            Division::div_rem_2_unnormalized::<true, false>(
                numerator.limbs(), low, high.max(1), &mut quotient, &mut sentinel,
            );
            prop_assert_eq!(&quotient, &expected);
            prop_assert_eq!(&sentinel, &InternalMpUint::from_limb(19));
            Division::div_rem_2_unnormalized::<false, true>(
                numerator.limbs(), low, high.max(1), &mut sentinel, &mut remainder,
            );
            prop_assert_eq!(&remainder, &residue);
            prop_assert_eq!(sentinel, InternalMpUint::from_limb(19));
        }
    }

    #[test]
    fn power_of_two_limb_division_preserves_exact_overlap(
        limbs in proptest::collection::vec(any::<Limb>(), 0..=129),
        shift in 0..Limb::BITS,
    ) {
        let numerator = InternalMpUint::from_limbs(limbs);
        let divisor = 1_usize << shift;
        let expected_remainder = numerator.limbs().first().copied().unwrap_or(0)
            & divisor.checked_sub(1).expect("positive power of two");
        let expected_quotient = numerator.shr(usize::try_from(shift).expect("native shift fits"));
        let mut quotient = InternalMpUint::from_limb(7);
        let remainder = Division::div_rem_1::<true>(numerator.limbs(), divisor, &mut quotient);
        prop_assert_eq!(&quotient, &expected_quotient);
        prop_assert_eq!(remainder, expected_remainder);
        let mut in_place = numerator.clone();
        prop_assert_eq!(Division::div_rem_1_assign(&mut in_place, divisor), expected_remainder);
        prop_assert_eq!(in_place, expected_quotient);
        let sentinel = quotient.clone();
        prop_assert_eq!(Division::div_rem_1::<false>(numerator.limbs(), divisor, &mut quotient),
            expected_remainder);
        prop_assert_eq!(quotient, sentinel);
    }

    #[test]
    fn reciprocal_two_by_one_matches_native_division(
        seed in prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()],
        high in prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()],
        low in prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()],
    ) {
        let divisor = seed | (1 << Limb::BITS.wrapping_sub(1));
        let u1 = high.checked_rem(divisor).expect("normalized divisor is nonzero");
        let numerator = (DoubleLimb::try_from(u1).expect("high limb fits") << Limb::BITS)
            | DoubleLimb::try_from(low).expect("low limb fits");
        let wide_divisor = DoubleLimb::try_from(divisor).expect("divisor fits");
        let inverse = Limb::try_from(
            DoubleLimb::MAX.div_euclid(wide_divisor)
                .checked_sub(DoubleLimb::from(1_u8) << Limb::BITS).expect("normalized inverse is nonnegative"),
        ).expect("normalized inverse fits one limb");
        let (quotient, remainder) = Division::divrem_2by1_reciprocal(
            u1, low, divisor, inverse,
        );
        prop_assert_eq!(quotient, Limb::try_from(numerator.div_euclid(wide_divisor)).expect("quotient fits"));
        prop_assert_eq!(remainder, Limb::try_from(numerator.rem_euclid(wide_divisor)).expect("remainder fits"));
    }

    #[test]
    fn reciprocal_three_by_two_recovers_constructed_quotients(
        high in any::<Limb>(),
        low in any::<Limb>(),
        quotient in prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()],
        remainder_high in any::<Limb>(),
        remainder_low in any::<Limb>(),
    ) {
        let d1 = high | (1 << Limb::BITS.wrapping_sub(1));
        let divisor = InternalMpUint::from_limbs_2(low, d1);
        let inverse = Division::invert_pi1(d1, low);
        for remainder in [
            InternalMpUint::zero(),
            divisor.sub(&InternalMpUint::one()),
            InternalMpUint::from_limbs_2(remainder_low,
                remainder_high.checked_rem(d1).expect("normalized divisor is nonzero")),
        ] {
            let numerator = divisor.mul(&InternalMpUint::from_limb(quotient)).add(&remainder);
            let limbs = numerator.limbs();
            let (actual_q, r1, r0) = Division::udiv_qr_3by2(
                limbs.get(2).copied().unwrap_or(0),
                limbs.get(1).copied().unwrap_or(0),
                limbs.first().copied().unwrap_or(0),
                d1, low, inverse,
            );
            prop_assert_eq!(actual_q, quotient);
            prop_assert_eq!(InternalMpUint::from_limbs_2(r0, r1), remainder);
        }
    }

    #[test]
    fn algorithm_d_initializes_complete_quotients_and_preserves_output_guards(
        mut denominator in proptest::collection::vec(any::<Limb>(), 2..=18),
        quotient in proptest::collection::vec(prop_oneof![Just(Limb::MAX), any::<Limb>()], 1..=20),
        write_quotient in any::<bool>(),
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1 << Limb::BITS.wrapping_sub(1);
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected_quotient = InternalMpUint::from_limbs(quotient);
        for expected_remainder in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
            let numerator = divisor.mul(&expected_quotient).add(&expected_remainder);
            let n = divisor.limbs().len();
            let mut dividend = numerator.limbs().to_vec();
            dividend.resize(dividend.len().max(n).checked_add(1).expect("test size fits"), 0);
            let m = dividend.len().checked_sub(n).and_then(|v| v.checked_sub(1)).expect("guarded dividend");
            let count = if write_quotient { m.checked_add(1).expect("test size fits") } else { 0 };
            let mut output = alloc::vec![7; count.checked_add(2).expect("guarded test output")];
            let mut remainder = alloc::vec![9; n];
            Division::knuth_d_divide_slice(
                &mut dividend, divisor.limbs(),
                output.get_mut(1..count.checked_add(1).expect("test size fits")).expect("output prefix"),
                &mut remainder,
            );
            prop_assert_eq!(output.first(), Some(&7));
            prop_assert_eq!(output.last(), Some(&7));
            let mut expected = expected_quotient.limbs().to_vec();
            expected.resize(count, 0);
            prop_assert_eq!(output.get(1..count.checked_add(1).expect("test size fits")).expect("output prefix"), expected);
            prop_assert_eq!(InternalMpUint::from_limbs(remainder), expected_remainder);
        }
    }
}

#[test]
fn two_limb_division_preserves_inline_quotients_at_every_shift() {
    for shift in 0..Limb::BITS {
        let divisor = InternalMpUint::from_limbs_2(Limb::MAX, 1 << shift);
        let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX, Limb::MAX, Limb::MAX, 1]);
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
            let numerator = divisor.mul(&expected).add(&residue);
            Division::div_rem_2_unnormalized::<true, true>(
                numerator.limbs(),
                Limb::MAX,
                1 << shift,
                &mut quotient,
                &mut remainder,
            );
            assert_eq!(quotient, expected);
            assert_eq!(quotient.capacity(), 4);
            assert_eq!(remainder, residue);
        }
    }
}

#[test]
fn two_limb_prefix_boundary_preserves_quotient_width_and_output_modes() {
    let boundary =
        InternalMpUint::one().shl(usize::try_from(Limb::BITS).expect("native width") * 4);
    for shift in 0..Limb::BITS {
        let divisor = InternalMpUint::from_limbs_2(Limb::MAX, 1 << shift);
        for expected in [
            boundary.sub(&InternalMpUint::one()),
            boundary.clone(),
            boundary.add(&InternalMpUint::one()),
        ] {
            for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
                let numerator = divisor.mul(&expected).add(&residue);
                let mut quotient = InternalMpUint::zero();
                let mut remainder = InternalMpUint::from_limb(17);
                Division::div_rem_2_unnormalized::<true, true>(
                    numerator.limbs(),
                    Limb::MAX,
                    1 << shift,
                    &mut quotient,
                    &mut remainder,
                );
                assert_eq!((&quotient, &remainder), (&expected, &residue));
                if expected.limbs().len() == 4 {
                    assert_eq!(quotient.capacity(), 4);
                }
                remainder = InternalMpUint::from_limb(17);
                Division::div_rem_2_unnormalized::<true, false>(
                    numerator.limbs(),
                    Limb::MAX,
                    1 << shift,
                    &mut quotient,
                    &mut remainder,
                );
                assert_eq!(quotient, expected);
                assert_eq!(remainder, InternalMpUint::from_limb(17));
                quotient = InternalMpUint::from_limb(19);
                Division::div_rem_2_unnormalized::<false, true>(
                    numerator.limbs(),
                    Limb::MAX,
                    1 << shift,
                    &mut quotient,
                    &mut remainder,
                );
                assert_eq!(quotient, InternalMpUint::from_limb(19));
                assert_eq!(remainder, residue);
            }
        }
    }
}

#[test]
fn single_limb_division_retains_normalized_remainders_for_every_shift() {
    let mut quotient = InternalMpUint::with_capacity(40);
    for shift in 0..Limb::BITS {
        for divisor in [1_usize << shift, Limb::MAX >> shift] {
            for width in [0, 1, 2, 3, 4, 5, 17, 33] {
                let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
                let mut expected = alloc::vec![0; width];
                let mut remainder = 0_u128;
                let divisor_wide = u128::try_from(divisor).expect("limb fits u128");
                for (source, target) in numerator.limbs().iter().zip(&mut expected).rev() {
                    let combined = (remainder << Limb::BITS)
                        | u128::try_from(*source).expect("limb fits u128");
                    *target = Limb::try_from(combined.div_euclid(divisor_wide))
                        .expect("quotient fits limb");
                    remainder = combined % divisor_wide;
                }
                let actual = Division::div_rem_1::<true>(numerator.limbs(), divisor, &mut quotient);
                assert_eq!(quotient, InternalMpUint::from_limbs(expected));
                assert_eq!(u128::try_from(actual).expect("limb fits"), remainder);
                let sentinel = quotient.clone();
                assert_eq!(
                    Division::div_rem_1::<false>(numerator.limbs(), divisor, &mut quotient),
                    actual
                );
                assert_eq!(quotient, sentinel);
            }
        }
    }
}
