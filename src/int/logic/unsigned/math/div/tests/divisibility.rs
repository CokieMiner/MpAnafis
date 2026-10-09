//! Destructive cancellation across stack, heap and power-of-two boundaries.

use proptest::prelude::*;

use crate::int::logic::unsigned::math::div::divisibility::{
    is_divisible_by_odd_limb, is_divisible_by_shifted_loop,
};

use super::{BURNIKEL_ZIEGLER_THRESHOLD, Division, InternalMpUint, LIMB_BITS, Limb};

#[test]
fn divisibility_zero_semantics_are_explicit() {
    let zero = InternalMpUint::zero();
    let nonzero = InternalMpUint::from_limb(17);
    assert!(zero.is_divisible_by(&zero));
    assert!(zero.is_divisible_by(&nonzero));
    assert!(!nonzero.is_divisible_by(&zero));
}

#[test]
fn equal_length_divisibility_handles_exact_and_near_multiples() {
    let mut limbs = alloc::vec![3; 6];
    *limbs.last_mut().expect("nonempty divisor") = 1;
    let divisor = InternalMpUint::from_limbs(limbs);
    let exact = divisor.mul(&InternalMpUint::from_limb(7));
    assert_eq!(exact.limbs().len(), divisor.limbs().len());
    for (value, expected) in [
        (exact.sub(&InternalMpUint::one()), false),
        (exact.clone(), true),
        (exact.add(&InternalMpUint::one()), false),
    ] {
        assert_eq!(value.is_divisible_by(&divisor), expected);
        assert_eq!(value.rem(&divisor).is_zero(), expected);
    }
}

#[test]
fn divisibility_removes_whole_and_partial_limb_power_of_two_factors() {
    let shift = LIMB_BITS + 3;
    let divisor = InternalMpUint::from_limbs(alloc::vec![3, 5, 1]).shl(shift);
    let unit = InternalMpUint::power_of_two(shift);
    let exact = divisor.mul(&InternalMpUint::from_limb(3));
    assert_eq!(exact.limbs().len(), divisor.limbs().len());
    assert!(exact.is_divisible_by(&divisor));
    assert!(exact.rem(&divisor).is_zero());
    assert!(!exact.add(&unit).is_divisible_by(&divisor));
    assert!(!exact.add(&unit).rem(&divisor).is_zero());
    let factor = InternalMpUint::from_limbs(alloc::vec![5, 0, 1]);
    let wide_exact = divisor.mul(&factor);
    assert!(wide_exact.limbs().len() > divisor.limbs().len());
    for (value, expected) in [
        (wide_exact.sub(&unit), false),
        (wide_exact.clone(), true),
        (wide_exact.add(&unit), false),
    ] {
        assert_eq!(value.is_divisible_by(&divisor), expected);
        assert_eq!(value.rem(&divisor).is_zero(), expected);
    }
}

#[test]
fn divisibility_matches_at_subquadratic_fallback_threshold_neighbors() {
    let mut limbs = alloc::vec![0; BURNIKEL_ZIEGLER_THRESHOLD];
    *limbs.first_mut().expect("nonempty divisor") = 3;
    *limbs.last_mut().expect("nonempty divisor") = 1;
    let divisor = InternalMpUint::from_limbs(limbs);
    for width in [BURNIKEL_ZIEGLER_THRESHOLD - 1, BURNIKEL_ZIEGLER_THRESHOLD] {
        let exact = divisor.shl(width * LIMB_BITS);
        assert_eq!(
            exact.limbs().len().checked_sub(divisor.limbs().len()),
            Some(width)
        );
        assert!(exact.is_divisible_by(&divisor));
        assert!(exact.rem(&divisor).is_zero());
        let mut near = exact;
        near.increment();
        assert!(!near.is_divisible_by(&divisor));
        assert!(!near.rem(&divisor).is_zero());
    }
}

#[test]
fn cancellation_keeps_the_guard_digit_at_equal_leading_limbs() {
    for width in [2_usize, 3, 4, 7, 31] {
        for high in [1, Limb::MAX >> 1, Limb::MAX] {
            let mut limbs = alloc::vec![Limb::MAX; width];
            *limbs.last_mut().expect("nonempty divisor") = high;
            let divisor = InternalMpUint::from_limbs(limbs);
            for factor in [
                InternalMpUint::from_limbs(alloc::vec![0, 1]),
                InternalMpUint::from_limb(Limb::MAX),
                InternalMpUint::from_limbs(alloc::vec![Limb::MAX, Limb::MAX]),
            ] {
                let exact = divisor.mul(&factor);
                for input in [
                    exact.sub(&InternalMpUint::one()),
                    exact.clone(),
                    exact.add(&InternalMpUint::one()),
                ] {
                    let mut cancelled = input.limbs().to_vec();
                    assert_eq!(
                        is_divisible_by_shifted_loop(&mut cancelled, divisor.limbs()),
                        input == exact,
                        "width={width}, high={high}, factor={factor:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn equal_high_limbs_classify_identity_and_adjacent_values() {
    for width in [2, 4, 5, 64, 65] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let below = divisor.sub(&InternalMpUint::from_limb(2));
        let mut same_high = divisor.limbs().to_vec();
        *same_high.first_mut().expect("nonempty divisor") = 3;
        let other = InternalMpUint::from_limbs(same_high);
        assert!(divisor.is_divisible_by(&divisor));
        assert!(!below.is_divisible_by(&divisor));
        assert!(!divisor.is_divisible_by(&other));
    }
}

#[test]
fn cancellation_covers_stack_edges_and_shifted_scalar_divisors() {
    for width in [
        1, 2, 3, 4, 5, 61, 62, 63, 64, 65, 66, 125, 126, 127, 128, 129,
    ] {
        for top in [1, Limb::MAX] {
            let mut limbs = alloc::vec![Limb::MAX; width];
            *limbs.last_mut().expect("positive divisor width") = top;
            let odd = InternalMpUint::from_limbs(limbs);
            for words in [0_usize, 1, 65] {
                for bits in [0, 1, LIMB_BITS.checked_sub(1).expect("positive limb width")] {
                    let shift = words
                        .checked_mul(LIMB_BITS)
                        .and_then(|v| v.checked_add(bits))
                        .expect("bounded test shift");
                    let divisor = odd.shl(shift);
                    let factor = InternalMpUint::from_limbs(alloc::vec![3, 5, 1]);
                    let exact = divisor.mul(&factor);
                    let unit = InternalMpUint::power_of_two(shift);
                    for input in [exact.sub(&unit), exact, divisor.mul(&factor).add(&unit)] {
                        assert_eq!(
                            input.is_divisible_by(&divisor),
                            input.rem(&divisor).is_zero(),
                            "width={width}, top={top}, shift={shift}"
                        );
                    }
                }
            }
        }
    }
}

proptest! {
    #[test]
    fn scalar_predicate_agrees_with_division(
        limbs in proptest::collection::vec(any::<Limb>(), 2..=140),
        seed in 1_usize..=Limb::MAX,
    ) {
        let divisor = seed | 3;
        let residue = Division::div_rem_1::<false>(&limbs, divisor, &mut InternalMpUint::zero());
        prop_assert_eq!(is_divisible_by_odd_limb(&limbs, divisor), residue == 0);
    }

    #[test]
    fn divisibility_agrees_with_remainder_for_shifted_odd_parts(
        mut digits in proptest::collection::vec(any::<Limb>(), 1..=140),
        quotient in proptest::collection::vec(any::<Limb>(), 1..=70),
        words in 0_usize..=80,
        bits in 0_usize..LIMB_BITS,
        remainder in 0_usize..=2,
    ) {
        *digits.first_mut().expect("nonempty divisor") |= 1;
        let odd = InternalMpUint::from_limbs(digits);
        let shift = words.checked_mul(LIMB_BITS).and_then(|v| v.checked_add(bits))
            .expect("bounded test shift");
        let divisor = odd.shl(shift);
        let factor = InternalMpUint::from_limbs(quotient);
        let input = divisor.mul(&factor).add(&InternalMpUint::from_limb(remainder).shl(shift));
        prop_assert_eq!(input.is_divisible_by(&divisor), input.rem(&divisor).is_zero());
    }
}
