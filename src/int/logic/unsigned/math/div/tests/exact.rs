//! Known-divisibility contracts, short-window correction, and output guards.

use proptest::prelude::*;

use super::{
    BURNIKEL_ZIEGLER_THRESHOLD, DIVISION_STACK_LIMBS, DivScratch, Division, InternalMpUint, Limb,
    NEWTON_RAPHSON_THRESHOLD, PreparedDivisor,
};

proptest! {
    #[test]
    fn exact_newton_recovers_constructed_quotients(
        denominator in proptest::collection::vec(any::<Limb>(), 1..=90),
        quotient_digits in proptest::collection::vec(any::<Limb>(), 0..=250),
        shift in 0_usize..=200,
    ) {
        let mut divisor = InternalMpUint::from_limbs(denominator);
        prop_assume!(!divisor.is_zero());
        divisor.shl_assign(shift);
        let expected = InternalMpUint::from_limbs(quotient_digits);
        let numerator = divisor.mul(&expected);
        let mut actual = InternalMpUint::from_limb(11);
        let mut discarded = InternalMpUint::zero();
        Division::newton::<true, false, true>(
            &numerator, &divisor, &mut actual, &mut discarded, &mut DivScratch::default(),
        );
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn exact_division_recovers_constructed_quotients(
        mut denominator in proptest::collection::vec(any::<Limb>(), 1..=160),
        digits in proptest::collection::vec(any::<Limb>(), 0..=160),
        scalar in 1..=Limb::MAX,
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1;
        let expected = InternalMpUint::from_limbs(digits);
        let wide = InternalMpUint::from_limbs(denominator);
        let mut output = InternalMpUint::from_limb(7);
        let mut scratch = DivScratch::default();
        for divisor in [wide, InternalMpUint::from_limb(scalar)] {
            let numerator = divisor.mul(&expected);
            Division::div_exact_into(&numerator, &divisor, &mut output, &mut scratch);
            prop_assert_eq!(&output, &expected);
        }
    }

    #[test]
    fn exact_short_division_preserves_quotients_and_guard_limbs(
        mut denominator in proptest::collection::vec(
            prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 3..=140,
        ),
        mut digits in proptest::collection::vec(
            prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=160,
        ),
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1 << Limb::BITS.wrapping_sub(1);
        *digits.last_mut().expect("nonempty quotient") |= 1;
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let numerator = divisor.mul(&expected);
        let length = numerator.limbs().len().checked_add(1).expect("test guard");
        let end = length.checked_add(1).expect("test sentinel");
        let mut storage = alloc::vec![17; end.checked_add(1).expect("bounded test allocation")];
        let window = storage.get_mut(1..end).expect("guarded numerator");
        window.fill(0);
        window.get_mut(..numerator.limbs().len()).expect("active numerator")
            .copy_from_slice(numerator.limbs());
        let q_len = length.checked_sub(divisor.limbs().len()).expect("positive quotient width");
        let q_end = q_len.checked_add(1).expect("quotient sentinel");
        let mut output = alloc::vec![23; q_end.checked_add(1).expect("bounded quotient allocation")];
        let quotient = output.get_mut(1..q_end).expect("guarded quotient");
        let prepared = PreparedDivisor::new(divisor.limbs());
        let certified = prepared.divide_quotient::<false, false, true>(
            window, divisor.limbs(), quotient,
        );
        prop_assert!(!certified);
        prop_assert_eq!(&InternalMpUint::from_limbs_slice(quotient), &expected);
        prop_assert_eq!((storage.first(), storage.last()), (Some(&17), Some(&17)));
        prop_assert_eq!((output.first(), output.last()), (Some(&23), Some(&23)));

        // The owner consumes the completed quotient without restoring inputs.
        let mut actual = InternalMpUint::from_limb(19);
        let mut untouched = InternalMpUint::from_limb(29);
        prop_assert!(!Division::algorithm_d::<true, false, false, true>(
            numerator.limbs(), divisor.limbs(), &mut actual, &mut untouched,
            &mut DivScratch::default(),
        ));
        prop_assert_eq!(actual, expected);
        prop_assert_eq!(untouched, InternalMpUint::from_limb(29));
    }
}

#[test]
fn newton_exact_quotients_cross_zero_and_limb_carries() {
    for width in [1, 2, 5, 21, 41] {
        let mut limbs = alloc::vec![Limb::MAX; width];
        *limbs.last_mut().expect("nonempty divisor") >>= 1;
        let den = InternalMpUint::from_limbs(limbs);
        let mut scratch = DivScratch::default();
        for quotient in [
            InternalMpUint::one(),
            InternalMpUint::from_limb(Limb::MAX),
            InternalMpUint::from_limbs_2(0, 1),
        ] {
            let num = den.mul(&quotient);
            let mut actual = InternalMpUint::zero();
            let mut rem = InternalMpUint::zero();
            Division::newton::<true, true, false>(&num, &den, &mut actual, &mut rem, &mut scratch);
            assert_eq!(actual, quotient, "exact quotient at width {width}");
            assert!(rem.is_zero());
        }
    }
}

#[test]
fn exact_newton_correction_recovers_every_admissible_error() {
    for shift in [0, 1, Limb::BITS - 1, Limb::BITS, Limb::BITS * 2 - 1] {
        for low in [1_usize, 3, Limb::MAX] {
            let mut divisor = InternalMpUint::from_limbs(alloc::vec![low, 1, 3]);
            divisor.shl_assign(usize::try_from(shift).expect("small shift"));
            for expected in [
                InternalMpUint::from_limb(2),
                InternalMpUint::from_limb(3),
                InternalMpUint::from_limb(Limb::MAX),
                InternalMpUint::from_limbs(alloc::vec![0, 1]),
                InternalMpUint::from_limbs(alloc::vec![0, 0, 0, 1]),
                InternalMpUint::from_limbs(alloc::vec![0, 0, 0, 0, 1]),
                InternalMpUint::from_limbs(alloc::vec![0, 0, 0, 0, 0, 1]),
            ] {
                let numerator = divisor.mul(&expected);
                let mut digits = numerator.limbs().to_vec();
                digits.resize(digits.len().max(divisor.limbs().len() + 1), 0);
                for error in 0..=3 {
                    let delta = InternalMpUint::from_limb(error);
                    if expected < delta {
                        continue;
                    }
                    let estimate = expected.sub(&delta);
                    let start = 3_usize;
                    let capacity = expected
                        .limbs()
                        .len()
                        .checked_add(start)
                        .and_then(|width| width.checked_add(1))
                        .expect("bounded quotient span");
                    let mut scratch = DivScratch::default();
                    scratch.v_padded.resize(capacity, 0);
                    scratch
                        .v_padded
                        .get_mut(..start)
                        .expect("product prefix")
                        .fill(7);
                    let end = start
                        .checked_add(estimate.limbs().len())
                        .expect("bounded estimate");
                    scratch
                        .v_padded
                        .get_mut(start..end)
                        .expect("estimate span")
                        .copy_from_slice(estimate.limbs());
                    Division::newton_exact_correction(
                        &digits,
                        divisor.limbs(),
                        start,
                        &mut scratch,
                    );
                    let actual = InternalMpUint::from_limbs_slice(
                        scratch.v_padded.get(start..).expect("quotient span"),
                    );
                    assert_eq!(actual, expected);
                    assert_eq!(scratch.v_padded.get(..start), Some([7, 7, 7].as_slice()));
                }
            }
        }
    }
}

#[test]
fn exact_scalar_division_covers_all_shifts_and_carry_boundaries() {
    let mut output = InternalMpUint::with_capacity(80);
    let mut scratch = DivScratch::default();
    for shift in 0..Limb::BITS {
        for odd in [1, 3, 5, Limb::MAX >> shift] {
            let Some(scalar) = odd.checked_shl(shift) else {
                continue;
            };
            if scalar >> shift != odd {
                continue;
            }
            let divisor = InternalMpUint::from_limb(scalar);
            for width in [0, 1, 3, 4, 5, 33] {
                for fill in [0, 1, Limb::MAX] {
                    let mut digits = alloc::vec![fill; width];
                    if let Some(high) = digits.last_mut() {
                        *high = Limb::MAX;
                    }
                    let expected = InternalMpUint::from_limbs(digits);
                    let numerator = divisor.mul(&expected);
                    Division::div_exact_into(&numerator, &divisor, &mut output, &mut scratch);
                    assert_eq!(output, expected);
                }
            }
        }
    }
}

#[test]
fn known_exact_dispatch_covers_normalization_and_crossovers() {
    let mut scratch = DivScratch::default();
    let mut actual = InternalMpUint::zero();
    for width in [
        3,
        4,
        5,
        DIVISION_STACK_LIMBS.checked_sub(1).expect("stack boundary"),
        DIVISION_STACK_LIMBS,
        DIVISION_STACK_LIMBS.checked_add(1).expect("stack boundary"),
        BURNIKEL_ZIEGLER_THRESHOLD
            .checked_sub(1)
            .expect("basecase boundary"),
        BURNIKEL_ZIEGLER_THRESHOLD,
        BURNIKEL_ZIEGLER_THRESHOLD
            .checked_add(1)
            .expect("recursive boundary"),
        NEWTON_RAPHSON_THRESHOLD
            .checked_sub(1)
            .expect("reciprocal boundary"),
        NEWTON_RAPHSON_THRESHOLD,
        NEWTON_RAPHSON_THRESHOLD
            .checked_add(1)
            .expect("reciprocal boundary"),
    ] {
        if cfg!(miri) && width > 65 {
            continue;
        }
        for top in [1, Limb::MAX >> 1, Limb::MAX] {
            let mut limbs = alloc::vec![Limb::MAX; width];
            *limbs.last_mut().expect("nonempty divisor") = top;
            let divisor = InternalMpUint::from_limbs(limbs);
            for q_len in [
                1,
                2,
                3,
                width >> 1,
                width,
                width.checked_mul(2).expect("bounded quotient"),
            ] {
                let quotient = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; q_len]);
                let numerator = divisor.mul(&quotient);
                Division::div_exact_into(&numerator, &divisor, &mut actual, &mut scratch);
                assert_eq!(
                    actual, quotient,
                    "divisor width {width}, quotient width {q_len}"
                );
            }
        }
    }
}

#[test]
fn overflowing_short_window_finishes_in_the_same_workspace() {
    let high = 1 << Limb::BITS.wrapping_sub(1);
    let divisor = InternalMpUint::from_limbs(alloc::vec![1, 0, high]);
    let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX, 0, 0, high]);
    let mut window = numerator.limbs().to_vec();
    let mut digits = [0];
    let prepared = PreparedDivisor::new(divisor.limbs());
    assert!(!prepared.divide_quotient::<false, false, false>(
        &mut window,
        divisor.limbs(),
        &mut digits,
    ),);
    assert_eq!(digits, [Limb::MAX]);
    assert_eq!(
        InternalMpUint::from_limbs_slice(window.get(..3).expect("three remainder limbs")),
        divisor.sub(&InternalMpUint::one())
    );
    let mut quotient = InternalMpUint::zero();
    let mut remainder = InternalMpUint::zero();
    let _ = Division::algorithm_d::<true, true, false, false>(
        numerator.limbs(),
        divisor.limbs(),
        &mut quotient,
        &mut remainder,
        &mut DivScratch::default(),
    );
    assert_eq!(quotient, InternalMpUint::from_limb(Limb::MAX));
    assert_eq!(remainder, divisor.sub(&InternalMpUint::one()));
}
