//! Equal-width scalar corrections and limb-wise power-of-two division.

use proptest::prelude::*;

use crate::int::types::INLINE_LIMBS;

use super::{DIVISION_SMALL_QUOTIENT_MAX, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn small_quotient_classifies_zero_and_one_from_leading_limbs() {
    let width = INLINE_LIMBS;
    let mut lower = alloc::vec![Limb::MAX; width];
    *lower.last_mut().expect("nonempty numerator") = 1;
    let mut upper = alloc::vec![0; width];
    *upper.last_mut().expect("nonempty divisor") = 2;
    let mut equal_low = alloc::vec![0; width];
    *equal_low.last_mut().expect("nonempty numerator") = 3;
    let mut equal_high = equal_low.clone();
    *equal_high.first_mut().expect("nonempty divisor") = 1;
    let mut equal_above = equal_high.clone();
    *equal_above.first_mut().expect("nonempty numerator") = 2;
    let mut different_above = alloc::vec![Limb::MAX; width];
    *different_above.last_mut().expect("nonempty numerator") = 3;
    for (num, den) in [
        (lower, upper.clone()),
        (equal_low, equal_high.clone()),
        (equal_above, equal_high),
        (different_above, upper),
    ] {
        let numerator = InternalMpUint::from_limbs(num);
        let divisor = InternalMpUint::from_limbs(den);
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        assert!(Division::small_quotient_div_rem(
            &numerator,
            &divisor,
            &mut quotient,
            &mut remainder
        ));
        let mut expected = InternalMpUint::zero();
        let mut residue = InternalMpUint::zero();
        let _ = Division::algorithm_d::<true, true, false, false>(
            numerator.limbs(),
            divisor.limbs(),
            &mut expected,
            &mut residue,
            &mut DivScratch::default(),
        );
        assert_eq!((quotient, remainder), (expected, residue));
    }
}

#[test]
fn small_quotient_handles_maximum_corrections_across_inline_boundary() {
    for width in [2, 3, INLINE_LIMBS, 5, 40] {
        for estimate in [DIVISION_SMALL_QUOTIENT_MAX - 1, DIVISION_SMALL_QUOTIENT_MAX] {
            // N=e*B^(n-1), D=2*B^(n-1)-1 realizes floor(e/2).
            let mut num = alloc::vec![0; width];
            *num.last_mut().expect("nonempty numerator") = estimate;
            let mut den = alloc::vec![Limb::MAX; width];
            *den.last_mut().expect("nonempty divisor") = 1;
            let numerator = InternalMpUint::from_limbs(num);
            let divisor = InternalMpUint::from_limbs(den);
            let mut quotient = InternalMpUint::zero();
            let mut remainder = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
            let capacity = remainder.capacity();
            let address = remainder.limbs().as_ptr();
            assert!(Division::small_quotient_div_rem(
                &numerator,
                &divisor,
                &mut quotient,
                &mut remainder
            ));
            assert_eq!(quotient, InternalMpUint::from_limb(estimate >> 1));
            assert_eq!(quotient.mul(&divisor).add(&remainder), numerator);
            assert!(remainder < divisor);
            assert_eq!(remainder.capacity(), capacity);
            assert_eq!(remainder.limbs().as_ptr(), address);
        }
    }
}

#[test]
fn small_quotient_absorbs_product_carry_without_a_guard_limb() {
    for width in [2, INLINE_LIMBS, 5, 64] {
        let mut num = alloc::vec![0; width];
        *num.last_mut().expect("nonempty numerator") = Limb::MAX;
        let mut den = alloc::vec![Limb::MAX; width];
        *den.last_mut().expect("nonempty divisor") = Limb::MAX.div_euclid(3);
        let numerator = InternalMpUint::from_limbs(num);
        let divisor = InternalMpUint::from_limbs(den);
        let expected = numerator.sub(&divisor.add(&divisor));
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        let admitted =
            Division::small_quotient_div_rem(&numerator, &divisor, &mut quotient, &mut remainder);
        assert_eq!(admitted, DIVISION_SMALL_QUOTIENT_MAX >= 3);
        if !admitted {
            assert!(quotient.is_zero() && remainder.is_zero());
            Division::div_rem_into(
                &numerator,
                &divisor,
                &mut quotient,
                &mut remainder,
                &mut DivScratch::default(),
            );
        }
        assert_eq!(quotient, InternalMpUint::from_limb(2));
        assert_eq!(remainder, expected);
        if admitted {
            assert_eq!(remainder.capacity(), width.max(INLINE_LIMBS));
        }
    }
}

#[test]
fn small_quotient_matches_algorithm_d_around_exact_multiple() {
    let mut den = alloc::vec![3; INLINE_LIMBS];
    *den.last_mut().expect("nonempty divisor") = 1;
    let divisor = InternalMpUint::from_limbs(den);
    let factor = InternalMpUint::from_limb(DIVISION_SMALL_QUOTIENT_MAX.min(5));
    let exact = divisor.mul(&factor);
    let one = InternalMpUint::one();
    for numerator in [exact.sub(&one), exact, divisor.mul(&factor).add(&one)] {
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        assert!(Division::small_quotient_div_rem(
            &numerator,
            &divisor,
            &mut quotient,
            &mut remainder
        ));
        let mut expected = InternalMpUint::zero();
        let mut residue = InternalMpUint::zero();
        let _ = Division::algorithm_d::<true, true, false, false>(
            numerator.limbs(),
            divisor.limbs(),
            &mut expected,
            &mut residue,
            &mut DivScratch::default(),
        );
        assert_eq!((quotient, remainder), (expected, residue));
    }
}

#[test]
fn small_quotient_rejection_leaves_outputs_untouched() {
    let divisor = InternalMpUint::from_limbs(alloc::vec![1; INLINE_LIMBS]);
    let factor = InternalMpUint::from_limb(DIVISION_SMALL_QUOTIENT_MAX).add(&InternalMpUint::one());
    let numerator = divisor.mul(&factor);
    let mut quotient = InternalMpUint::from_limb(0xdead);
    let mut remainder = InternalMpUint::from_limb(0xbeef);
    assert!(!Division::small_quotient_div_rem(
        &numerator,
        &divisor,
        &mut quotient,
        &mut remainder
    ));
    assert_eq!(quotient, InternalMpUint::from_limb(0xdead));
    assert_eq!(remainder, InternalMpUint::from_limb(0xbeef));
    let _ = Division::algorithm_d::<true, true, false, false>(
        numerator.limbs(),
        divisor.limbs(),
        &mut quotient,
        &mut remainder,
        &mut DivScratch::default(),
    );
    assert_eq!(quotient, factor);
    assert!(remainder.is_zero());
}

proptest! {
    #[test]
    fn doubling_comparison_covers_carries_and_adjacent_values(
        mut limbs in proptest::collection::vec(
            prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=129,
        ),
    ) {
        *limbs.last_mut().expect("nonempty divisor") |= 1;
        let divisor = InternalMpUint::from_limbs(limbs);
        let doubled = divisor.add(&divisor);
        let one = InternalMpUint::one();
        for numerator in [doubled.sub(&one), doubled.clone(), doubled.add(&one)] {
            if numerator.limbs().len() == divisor.limbs().len() {
                prop_assert_eq!(
                    Division::less_than_double(numerator.limbs(), divisor.limbs()),
                    numerator < doubled,
                );
            }
        }
        let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; divisor.limbs().len()]);
        prop_assert_eq!(
            Division::less_than_double(numerator.limbs(), divisor.limbs()),
            numerator < doubled,
        );
    }

    #[test]
    fn prop_small_quotient_addbacks_match_algorithm_d(
        low in proptest::collection::vec(any::<Limb>(), 1..=40),
        top in 1_usize..=8,
        multiple in 2_usize..=DIVISION_SMALL_QUOTIENT_MAX.min(8),
    ) {
        let mut den = low.clone();
        den.push(top);
        let divisor = InternalMpUint::from_limbs(den);
        let mut num = low;
        num.push(top.checked_mul(multiple).expect("leading limb is at most 64"));
        let numerator = InternalMpUint::from_limbs(num);
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        prop_assert!(Division::small_quotient_div_rem(&numerator, &divisor, &mut quotient, &mut remainder));
        let mut expected = InternalMpUint::zero();
        let mut residue = InternalMpUint::zero();
        let _ = Division::algorithm_d::<true, true, false, false>(
            numerator.limbs(), divisor.limbs(), &mut expected, &mut residue, &mut DivScratch::default(),
        );
        prop_assert_eq!((quotient, remainder), (expected, residue));
    }

    #[test]
    fn power_of_two_division_agrees_across_output_and_assignment_modes(
        mut limbs in proptest::collection::vec(any::<Limb>(), 1..=160),
        whole_seed in 0_usize..=159,
        shift in 0_usize..LIMB_BITS,
        output_len in prop_oneof![Just(0_usize), Just(INLINE_LIMBS), Just(INLINE_LIMBS + 1), 0_usize..=170],
    ) {
        *limbs.last_mut().expect("nonempty numerator") |= 1;
        let whole = whole_seed.min(limbs.len().checked_sub(1).expect("nonempty numerator"));
        let bits = whole.checked_mul(LIMB_BITS).and_then(|v| v.checked_add(shift)).expect("bounded exponent");
        let divisor = InternalMpUint::power_of_two(bits);
        let numerator = InternalMpUint::from_limbs(limbs);
        let expected_q = numerator.shr(bits);
        let expected_r = numerator.bitand(&divisor.sub(&InternalMpUint::one()));
        let mut quotient = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; output_len]);
        let mut remainder = quotient.clone();
        let q_capacity = quotient.capacity();
        let r_capacity = remainder.capacity();
        let q_pointer = quotient.limbs().as_ptr();
        let r_pointer = remainder.limbs().as_ptr();
        prop_assert!(Division::power_of_two::<true, true>(&numerator, &divisor, &mut quotient, &mut remainder));
        prop_assert_eq!((&quotient, &remainder), (&expected_q, &expected_r));
        if numerator.limbs().len().checked_sub(whole).expect("admitted suffix") <= q_capacity {
            prop_assert_eq!(quotient.limbs().as_ptr(), q_pointer);
        }
        if whole.checked_add(usize::from(shift != 0)).expect("bounded remainder width") <= r_capacity {
            prop_assert_eq!(remainder.limbs().as_ptr(), r_pointer);
        }
        let mut sentinel = InternalMpUint::from_limb(17);
        prop_assert!(Division::power_of_two::<true, false>(&numerator, &divisor, &mut quotient, &mut sentinel));
        prop_assert_eq!(&quotient, &expected_q);
        prop_assert_eq!(&sentinel, &InternalMpUint::from_limb(17));
        prop_assert!(Division::power_of_two::<false, true>(&numerator, &divisor, &mut sentinel, &mut remainder));
        prop_assert_eq!(&remainder, &expected_r);
        prop_assert_eq!(sentinel, InternalMpUint::from_limb(17));
        prop_assert_eq!(numerator.div_rem(&divisor), (expected_q.clone(), expected_r.clone()));
        let mut assigned = numerator.clone();
        assigned.div_assign(&divisor);
        prop_assert_eq!(assigned, expected_q);
        let mut assigned_remainder = numerator;
        assigned_remainder.rem_assign(&divisor);
        prop_assert_eq!(assigned_remainder, expected_r);
    }
}

#[test]
fn power_of_two_rejection_preserves_both_outputs() {
    let divisor = InternalMpUint::from_limbs(alloc::vec![1, 0, 1]);
    let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 4]);
    let mut quotient = InternalMpUint::from_limb(17);
    let mut remainder = InternalMpUint::from_limb(19);
    assert!(!Division::power_of_two::<true, true>(
        &numerator,
        &divisor,
        &mut quotient,
        &mut remainder
    ));
    assert_eq!(quotient, InternalMpUint::from_limb(17));
    assert_eq!(remainder, InternalMpUint::from_limb(19));
}
