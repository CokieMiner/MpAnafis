//! Streamed small-divisor division across normalization and output modes.

use core::num::NonZeroUsize;

use proptest::prelude::*;

use crate::int::logic::unsigned::INLINE_LIMBS;

use super::{
    CompleteDivisionMode, DIVISION_STACK_LIMBS, DivScratch, Division, DivisionAttempt,
    InternalMpUint, Limb, QuotientAndRemainder, QuotientOnly, RemainderOnly,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 8 } else { 128 }))]

    #[test]
    fn streaming_division_preserves_all_output_contracts(
        lower_words in proptest::array::uniform7(
            prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()],
        ),
        high_seed in any::<Limb>(),
        shift in 0_u32..Limb::BITS,
        quotient_words in proptest::collection::vec(
            prop_oneof![Just(0), Just(1), Just(Limb::MAX), any::<Limb>()],
            0..=if cfg!(miri) { 5 } else { DIVISION_STACK_LIMBS + 2 },
        ),
        residue_words in any::<[Limb; 8]>(),
        residue_kind in 0_u8..4,
        padding in 0_usize..=3,
    ) {
        let high_bit = Limb::BITS.checked_sub(1).and_then(|bits| bits.checked_sub(shift))
            .expect("normalization shift is below limb width");
        let high = (high_seed >> shift) | (1 << high_bit);
        let quotient = InternalMpUint::from_limbs(quotient_words);
        for width in 3_usize..=8 {
            let low_count = width.checked_sub(1).expect("fixed width is positive");
            let mut words = lower_words.get(..low_count).expect("fixed divisor prefix").to_vec();
            words.push(high);
            let divisor = InternalMpUint::from_limbs(words.clone());
            let remainder = match residue_kind {
                0 => InternalMpUint::zero(),
                1 => InternalMpUint::one(),
                2 => divisor.sub(&InternalMpUint::one()),
                _ => {
                    let mut limbs = residue_words.get(..low_count)
                        .expect("fixed residue prefix").to_vec();
                    limbs.push(residue_words.last().copied().expect("nonempty residue words").rem_euclid(high));
                    InternalMpUint::from_limbs(limbs)
                },
            };
            check_case(&words, &quotient, &remainder, padding);
        }
    }
}

#[test]
fn leading_pair_saturation_and_low_product_addback_are_exact() {
    let radix = InternalMpUint::from_limbs(alloc::vec![0, 1]);
    let maximum_digit = InternalMpUint::from_limb(Limb::MAX);
    let one = InternalMpUint::one();
    for shift in 0..Limb::BITS {
        let high_bit = Limb::BITS
            .checked_sub(1)
            .and_then(|bits| bits.checked_sub(shift))
            .expect("normalization shift is below limb width");
        for low in [1, Limb::MAX] {
            for width in 3_usize..=8 {
                let mut words = alloc::vec![0; width];
                *words.first_mut().expect("fixed divisor bottom") = low;
                *words.last_mut().expect("fixed divisor high") = 1 << high_bit;
                for second in [0, Limb::MAX] {
                    *words.get_mut(1).expect("fixed divisor second word") = second;
                    let divisor = InternalMpUint::from_limbs(words.clone());
                    let largest_remainder = divisor.sub(&one);
                    // D*B-1 gives digit B-1 and remainder D-1.
                    check_case(&words, &maximum_digit, &largest_remainder, 0);
                    // 2*D-1 requires correcting an estimate that omits low limbs.
                    check_case(&words, &one, &largest_remainder, 0);
                    // Appending a quotient limb exercises the ordinary digit loop.
                    check_case(&words, &radix, &InternalMpUint::zero(), 1);
                }
            }
        }
    }
    let top_bit = Limb::BITS.checked_sub(1).expect("limb width is positive");
    for width in 4_usize..=8 {
        let low_count = width.checked_sub(2).expect("fixed leading pair");
        let mut words = alloc::vec![Limb::MAX; low_count];
        words.extend([0, 1 << top_bit]);
        let divisor = InternalMpUint::from_limbs(words.clone());
        let maximum_prefix = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; low_count]);
        // Matching leading pairs give Rprev=D-delta. delta=B exercises
        // a zero bottom gap and carry from incoming+d0; the maximal prefix
        // exercises the full low-word borrow chain.
        for delta in [&radix, &maximum_prefix] {
            let remainder = divisor.sub(&delta.mul(&radix)).add(&maximum_digit);
            check_case(&words, &maximum_digit, &remainder, 0);
        }
    }
}

fn check_case(
    words: &[Limb],
    expected_quotient: &InternalMpUint,
    expected_remainder: &InternalMpUint,
    padding: usize,
) {
    let divisor = InternalMpUint::from_limbs(words.to_vec());
    assert!(expected_remainder < &divisor);
    let numerator = expected_quotient.mul(&divisor).add(expected_remainder);
    let mut input = numerator.limbs().to_vec();
    let length = input
        .len()
        .max(words.len())
        .checked_add(padding)
        .expect("bounded test input");
    input.resize(length, 0);
    let before = input.clone();
    let sentinel = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; length.max(5)]);
    let mut quotient = sentinel.clone();
    let mut remainder = sentinel.clone();
    let quotient_storage = quotient.limbs().as_ptr();
    let remainder_storage = remainder.limbs().as_ptr();
    divide::<QuotientAndRemainder>(&input, words, &mut quotient, &mut remainder);
    assert_eq!(&quotient, expected_quotient);
    assert_eq!(&remainder, expected_remainder);
    assert_eq!(quotient.limbs().as_ptr(), quotient_storage);
    assert_eq!(remainder.limbs().as_ptr(), remainder_storage);

    quotient.clone_from(&sentinel);
    remainder.clone_from(&sentinel);
    divide::<QuotientOnly>(&input, words, &mut quotient, &mut remainder);
    assert_eq!(&quotient, expected_quotient);
    assert_eq!(remainder, sentinel);

    quotient.clone_from(&sentinel);
    divide::<RemainderOnly>(&input, words, &mut quotient, &mut remainder);
    assert_eq!(quotient, sentinel);
    assert_eq!(&remainder, expected_remainder);

    let mut fresh_remainder = InternalMpUint::zero();
    divide::<RemainderOnly>(&input, words, &mut quotient, &mut fresh_remainder);
    assert_eq!(&fresh_remainder, expected_remainder);
    if expected_remainder.limbs().len() <= INLINE_LIMBS {
        assert_eq!(fresh_remainder.capacity(), INLINE_LIMBS);
    }

    if expected_remainder.is_zero() {
        divide::<QuotientOnly<true>>(&input, words, &mut quotient, &mut remainder);
        assert_eq!(&quotient, expected_quotient);
    }
    assert_eq!(input, before);

    if numerator.limbs().len() >= words.len() {
        let mut scratch = DivScratch::default();
        Division::algorithm_d::<QuotientAndRemainder>(
            numerator.limbs(),
            divisor.limbs(),
            &mut quotient,
            &mut remainder,
            &mut scratch,
        );
        assert_eq!(&quotient, expected_quotient);
        assert_eq!(&remainder, expected_remainder);
        assert!(scratch.u_norm.is_empty() && scratch.v_norm.is_empty());
        assert_eq!(
            Division::try_algorithm_d_unscratched::<QuotientOnly, false>(
                &numerator,
                &divisor,
                &mut quotient,
                &mut remainder,
            ),
            DivisionAttempt::Complete
        );
        assert_eq!(&quotient, expected_quotient);
    }
}

fn divide<Mode: CompleteDivisionMode>(
    numerator: &[Limb],
    words: &[Limb],
    quotient: &mut InternalMpUint,
    remainder: &mut InternalMpUint,
) {
    if let [low, middle, high] = *words {
        Division::div_rem_3_unnormalized::<Mode>(
            numerator,
            (
                low,
                middle,
                NonZeroUsize::new(high).expect("nonzero high limb"),
            ),
            quotient,
            remainder,
        );
    } else if words.len() == 4 {
        let [bottom, low, middle, high] = <[Limb; 4]>::try_from(words).expect("four-limb fixture");
        Division::div_rem_4_unnormalized::<Mode>(
            numerator,
            (
                bottom,
                low,
                middle,
                NonZeroUsize::new(high).expect("nonzero high limb"),
            ),
            quotient,
            remainder,
        );
    } else {
        Division::div_rem_fixed_unnormalized::<Mode>(numerator, words, quotient, remainder);
    }
}
