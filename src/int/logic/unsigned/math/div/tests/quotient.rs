//! Prefix certificates and corrections around exact products.

use proptest::prelude::*;

use crate::int::logic::unsigned::math::div::quotient::truncated_quotient_into;

use super::{
    BURNIKEL_LONG_QUOTIENT_THRESHOLD, BURNIKEL_QUOTIENT_THRESHOLD, BURNIKEL_ZIEGLER_THRESHOLD,
    DivScratch, Division, InternalMpUint, LIMB_BITS, Limb, NEWTON_QUOTIENT_THRESHOLD,
    NEWTON_RAPHSON_THRESHOLD,
};

proptest! {
    #[test]
    fn scalar_truncation_corrects_products_without_product_storage(
        mut denominator in proptest::collection::vec(any::<Limb>(), 8..=160),
        digit in 2_usize..Limb::MAX,
        leading in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
    ) {
        *denominator.last_mut().expect("nonempty divisor") = leading.max(1);
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limb(digit);
        let one = InternalMpUint::one();
        let product = divisor.mul(&expected);
        let mut scratch = DivScratch::default();
        for (numerator, quotient) in [
            (product.sub(&one), expected.sub(&one)),
            (product.clone(), expected.clone()),
            (product.add(&one), expected.clone()),
            (product.add(&divisor.sub(&one)), expected),
        ] {
            let mut output = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 6]);
            prop_assert!(Division::truncated_quotient::<false>(&numerator, &divisor, &mut output));
            prop_assert_eq!(&output, &quotient);
            prop_assert!(truncated_quotient_into::<false>(
                &numerator, &divisor, &mut output, &mut scratch,
            ));
            prop_assert_eq!(&output, &quotient);
            prop_assert_eq!(scratch.q_den_low.capacity(), 0);
        }
    }

    #[test]
    fn recursive_quotient_guards_cover_normalization_and_ambiguous_residues(
        mut denominator in proptest::collection::vec(any::<Limb>(),
            if cfg!(miri) { 24..=32 } else { 192..=384 }),
        mut digits in proptest::collection::vec(any::<Limb>(),
            if cfg!(miri) { 4..=8 } else { 96..=128 }),
        leading in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
    ) {
        *denominator.last_mut().expect("nonempty divisor") = leading.max(1);
        *digits.last_mut().expect("nonempty quotient") |= 1;
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let mut quotient = InternalMpUint::zero();
        let mut scratch = DivScratch::default();
        for residue in [
            InternalMpUint::zero(), InternalMpUint::one(),
            divisor.shr(LIMB_BITS), divisor.shr(LIMB_BITS - 1),
            divisor.shr(1), divisor.sub(&InternalMpUint::one()),
        ] {
            let numerator = product.add(&residue);
            let prior = quotient.clone();
            if !truncated_quotient_into::<false>(&numerator, &divisor, &mut quotient, &mut scratch) {
                prop_assert_eq!(&quotient, &prior);
                Division::div_into::<false, false>(&numerator, &divisor, &mut quotient, &mut scratch);
            }
            prop_assert_eq!(&quotient, &expected);
            let retained = numerator.limbs().len().checked_sub(divisor.limbs().len())
                .and_then(|length| length.checked_add(2)).expect("bounded prefix width");
            let split = divisor.limbs().len().checked_sub(retained).expect("shorter prefix");
            Division::guarded_quotient::<false>(numerator.limbs(), divisor.limbs(), &mut quotient, split, &mut scratch);
            prop_assert_eq!(&quotient, &expected);
        }
        let below = product.sub(&InternalMpUint::one());
        let prior = quotient.clone();
        if !truncated_quotient_into::<false>(&below, &divisor, &mut quotient, &mut scratch) {
            prop_assert_eq!(&quotient, &prior);
            Division::div_into::<false, false>(&below, &divisor, &mut quotient, &mut scratch);
        }
        prop_assert_eq!(quotient, expected.sub(&InternalMpUint::one()));
    }

    #[test]
    fn truncated_prefix_certificate_preserves_the_exact_quotient(
        mut denominator in proptest::collection::vec(any::<Limb>(), 3..=80),
        mut numerator in proptest::collection::vec(any::<Limb>(), 158),
        leading in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
    ) {
        *denominator.last_mut().expect("nonempty divisor") = leading.max(1);
        numerator.truncate(denominator.len().checked_mul(2).and_then(|width| width.checked_sub(2)).expect("bounded prefix width"));
        *numerator.last_mut().expect("nonempty numerator") |= 1;
        let divisor = InternalMpUint::from_limbs(denominator);
        let dividend = InternalMpUint::from_limbs(numerator);
        let (expected, remainder) = dividend.div_rem(&divisor);
        let mut quotient = InternalMpUint::zero();
        let mut output_remainder = InternalMpUint::from_limb(17);
        let certified = Division::algorithm_d::<true, true, true, false>(
            dividend.limbs(), divisor.limbs(), &mut quotient, &mut output_remainder,
            &mut DivScratch::default(),
        );
        prop_assert_eq!(&quotient, &expected);
        if certified {
            prop_assert!(remainder > quotient);
            prop_assert_eq!(output_remainder, InternalMpUint::from_limb(17));
        } else {
            prop_assert_eq!(output_remainder, remainder);
        }
    }

    #[test]
    fn truncated_prefix_handles_unnormalized_divisors_and_residues(
        mut denominator in proptest::collection::vec(any::<Limb>(), 12..=80),
        digits in proptest::collection::vec(any::<Limb>(), 1..=4),
        leading in prop_oneof![Just(1), Just(2), Just(Limb::MAX), any::<Limb>()],
        residue in any::<Limb>(),
    ) {
        *denominator.last_mut().expect("nonempty divisor") = leading.max(1);
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let mut output = InternalMpUint::zero();
        for remainder in [
            InternalMpUint::zero(), InternalMpUint::from_limb(residue),
            divisor.shr(1), divisor.sub(&InternalMpUint::one()),
        ] {
            let numerator = product.add(&remainder);
            let prior = output.clone();
            if !Division::truncated_quotient::<false>(&numerator, &divisor, &mut output) {
                prop_assert_eq!(&output, &prior);
                Division::div_into::<false, false>(&numerator, &divisor, &mut output, &mut DivScratch::default());
            }
            prop_assert_eq!(&output, &expected);
        }
        if !product.is_zero() {
            let numerator = product.sub(&InternalMpUint::one());
            let prior = output.clone();
            if !Division::truncated_quotient::<false>(&numerator, &divisor, &mut output) {
                prop_assert_eq!(&output, &prior);
                Division::div_into::<false, false>(&numerator, &divisor, &mut output, &mut DivScratch::default());
            }
            prop_assert_eq!(output, expected.sub(&InternalMpUint::one()));
        }
    }

    #[test]
    fn balanced_quotients_preserve_identity_through_recursive_certificates(
        mut denominator in proptest::collection::vec(any::<Limb>(), 128..=320),
        mut digits in proptest::collection::vec(any::<Limb>(), 320),
        leading in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
    ) {
        *denominator.last_mut().expect("nonempty divisor") = leading.max(1);
        digits.truncate(denominator.len());
        *digits.last_mut().expect("nonempty quotient") |= 1;
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let mut quotient = InternalMpUint::zero();
        let mut scratch = DivScratch::default();
        for residue in [InternalMpUint::zero(), divisor.shr(1), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&residue);
            Division::div_into::<true, false>(&numerator, &divisor, &mut quotient, &mut scratch);
            prop_assert_eq!(&quotient, &expected);
            prop_assert_eq!(numerator.div(&divisor), expected.clone());
        }
    }

    #[test]
    fn half_width_quotients_cover_exact_and_adjacent_products(
        mut limbs in proptest::collection::vec(any::<Limb>(),
            BURNIKEL_ZIEGLER_THRESHOLD..=BURNIKEL_ZIEGLER_THRESHOLD * 2 - 8),
        leading in prop_oneof![Just(1), Just(Limb::MAX), any::<Limb>()],
    ) {
        let width = limbs.len();
        *limbs.last_mut().expect("nonempty divisor") = leading.max(1);
        let divisor = InternalMpUint::from_limbs(limbs);
        let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width >> 1]);
        let product = divisor.mul(&expected);
        for residue in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
            prop_assert_eq!(product.add(&residue).div(&divisor), expected.clone());
        }
        prop_assert_eq!(product.sub(&InternalMpUint::one()).div(&divisor), expected.sub(&InternalMpUint::one()));
    }

    #[test]
    fn prop_truncated_quotient_matches_full_division(
        mut den in proptest::collection::vec(any::<Limb>(), 8..=40),
        extra in proptest::collection::vec(any::<Limb>(), 0..=6),
    ) {
        *den.last_mut().expect("nonempty divisor") |= 1;
        let divisor = InternalMpUint::from_limbs(den.clone());
        den.extend_from_slice(&extra);
        let numerator = InternalMpUint::from_limbs(den);
        let mut quotient = InternalMpUint::zero();
        if Division::truncated_quotient::<false>(&numerator, &divisor, &mut quotient) {
            prop_assert_eq!(quotient, numerator.div_rem(&divisor).0);
        }
    }

    #[test]
    fn prop_truncated_quotient_matches_near_exact_multiples(
        mut den in proptest::collection::vec(any::<Limb>(), 8..=40),
        factor in proptest::collection::vec(any::<Limb>(), 1..=3),
        offset in any::<u8>(),
    ) {
        *den.last_mut().expect("nonempty divisor") |= 1;
        let divisor = InternalMpUint::from_limbs(den);
        let product = divisor.mul(&InternalMpUint::from_limbs(factor));
        let numerator = product.add(&InternalMpUint::from_limb(Limb::from(offset)));
        let mut quotient = InternalMpUint::zero();
        if Division::truncated_quotient::<false>(&numerator, &divisor, &mut quotient) {
            prop_assert_eq!(quotient, numerator.div_rem(&divisor).0);
        }
    }

    #[test]
    fn prop_equal_width_truncated_quotient_matches_full_division(
        mut den in proptest::collection::vec(any::<Limb>(), 2..=40),
        multiple in any::<u8>(),
        tail in any::<Limb>(),
    ) {
        *den.last_mut().expect("nonempty divisor") |= 1;
        let divisor = InternalMpUint::from_limbs(den);
        let numerator = divisor.mul(&InternalMpUint::from_limb(Limb::from(multiple)))
            .add(&InternalMpUint::from_limb(tail));
        let mut quotient = InternalMpUint::zero();
        if Division::truncated_quotient::<false>(&numerator, &divisor, &mut quotient) {
            prop_assert_eq!(quotient, numerator.div_rem(&divisor).0);
        }
    }
}

#[test]
fn scalar_truncation_decrements_a_heap_quotient_to_one_without_normalization() {
    let mut limbs = alloc::vec![Limb::MAX; 8];
    *limbs.last_mut().expect("nonempty divisor") = 1;
    let divisor = InternalMpUint::from_limbs(limbs);
    let numerator = divisor
        .mul(&InternalMpUint::from_limb(2))
        .sub(&InternalMpUint::one());
    let mut quotient = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 6]);
    let capacity = quotient.capacity();
    assert!(Division::truncated_quotient::<false>(
        &numerator,
        &divisor,
        &mut quotient,
    ));
    assert_eq!(quotient, InternalMpUint::one());
    assert_eq!(quotient.limbs(), &[1]);
    assert_eq!(quotient.capacity(), capacity);
}

#[test]
fn guarded_quotient_handles_zero_padded_discarded_prefixes() {
    let width = (BURNIKEL_ZIEGLER_THRESHOLD * 2).max(8);
    let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
    let mut scratch = DivScratch::default();
    for low in [0, 1, Limb::MAX] {
        let mut limbs = alloc::vec![0; width * 2];
        *limbs.first_mut().expect("nonempty divisor") = low;
        *limbs.last_mut().expect("nonempty divisor") = 1;
        let divisor = InternalMpUint::from_limbs(limbs);
        let product = divisor.mul(&expected);
        let mut quotient = InternalMpUint::zero();
        for residue in [
            InternalMpUint::zero(),
            InternalMpUint::one(),
            divisor.sub(&InternalMpUint::one()),
        ] {
            let numerator = product.add(&residue);
            let prior = quotient.clone();
            if !truncated_quotient_into::<false>(&numerator, &divisor, &mut quotient, &mut scratch)
            {
                assert_eq!(quotient, prior);
                Division::div_into::<false, false>(
                    &numerator,
                    &divisor,
                    &mut quotient,
                    &mut scratch,
                );
            }
            assert_eq!(quotient, expected);
            assert_eq!(numerator.div(&divisor), expected);
        }
        assert_eq!(
            product.sub(&InternalMpUint::one()).div(&divisor),
            expected.sub(&InternalMpUint::one())
        );
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Production quotient guards include the large Newton crossover; bounded guard-certificate properties run under Miri."
)]
fn quotient_guard_crossovers_preserve_exact_and_adjacent_products() {
    let one = InternalMpUint::one();
    for crossover in [
        (BURNIKEL_ZIEGLER_THRESHOLD * 2).max(4),
        (BURNIKEL_ZIEGLER_THRESHOLD * 3).max(4),
        NEWTON_RAPHSON_THRESHOLD.max(4),
    ] {
        for width in [crossover - 1, crossover, crossover + 1] {
            let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width * 2]);
            let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width - 2]);
            let product = divisor.mul(&expected);
            for residue in [InternalMpUint::zero(), one.clone(), divisor.sub(&one)] {
                assert_eq!(product.add(&residue).div(&divisor), expected);
            }
            assert_eq!(product.sub(&one).div(&divisor), expected.sub(&one));
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The full Burnikel crossover matrix repeats wide quotient-only divisions; bounded quotient-policy and guard properties run under Miri"
)]
fn quotient_only_crossovers_preserve_exact_and_adjacent_products() {
    let crossover = BURNIKEL_ZIEGLER_THRESHOLD
        .checked_mul(3)
        .expect("bounded division crossover")
        .max(4);
    let one = InternalMpUint::one();
    let mut quotient = InternalMpUint::zero();
    let mut scratch = DivScratch::default();
    let recursive_crossover = crossover
        .checked_mul(2)
        .and_then(|width| width.checked_sub(2))
        .expect("bounded recursive crossover");
    for boundary in [crossover, recursive_crossover] {
        for width in [boundary - 1, boundary, boundary + 1] {
            let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
            for leading in [1, Limb::MAX] {
                let mut limbs = alloc::vec![Limb::MAX; width];
                *limbs.last_mut().expect("nonempty divisor") = leading;
                let divisor = InternalMpUint::from_limbs(limbs);
                let product = divisor.mul(&expected);
                for residue in [InternalMpUint::zero(), one.clone(), divisor.sub(&one)] {
                    let numerator = product.add(&residue);
                    Division::div_into::<true, false>(
                        &numerator,
                        &divisor,
                        &mut quotient,
                        &mut scratch,
                    );
                    assert_eq!(quotient, expected);
                    assert_eq!(numerator.div(&divisor), expected);
                }
                assert_eq!(product.sub(&one).div(&divisor), expected.sub(&one));
            }
        }
    }
}

#[test]
fn truncated_prefix_reuses_exact_remainders_when_certification_is_ambiguous() {
    for width in [3, 4, 16, 32, 33, 64, 95, 96] {
        for leading in [1, 1 << Limb::BITS.wrapping_sub(1)] {
            let mut limbs = alloc::vec![0; width];
            *limbs.last_mut().expect("nonempty divisor") = leading;
            let divisor = InternalMpUint::from_limbs(limbs);
            let mut digits = alloc::vec![0; width - 1];
            *digits.last_mut().expect("nonempty quotient") = 1;
            let expected = InternalMpUint::from_limbs(digits);
            let product = divisor.mul(&expected);
            for (remainder, certified) in [
                (InternalMpUint::zero(), false),
                (InternalMpUint::one(), false),
                (divisor.shr(1), true),
            ] {
                let dividend = product.add(&remainder);
                assert_eq!(dividend.limbs().len(), 2 * width - 2);
                let mut quotient = InternalMpUint::from_limb(91);
                let mut output_remainder = InternalMpUint::from_limb(17);
                let certificate = Division::algorithm_d::<true, true, true, false>(
                    dividend.limbs(),
                    divisor.limbs(),
                    &mut quotient,
                    &mut output_remainder,
                    &mut DivScratch::default(),
                );
                assert_eq!(quotient, expected);
                assert_eq!(certificate, certified);
                if certified {
                    assert!(remainder > quotient);
                    assert_eq!(output_remainder, InternalMpUint::from_limb(17));
                } else {
                    assert_eq!(output_remainder, remainder);
                }
            }
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The discarded-product matrix includes production Newton crossovers; bounded prefix-certificate and truncated-quotient properties run under Miri"
)]
fn truncated_quotient_corrects_discarded_products_at_recursive_boundaries() {
    let mut output = InternalMpUint::zero();
    for (width, guards) in [
        BURNIKEL_ZIEGLER_THRESHOLD,
        BURNIKEL_QUOTIENT_THRESHOLD,
        BURNIKEL_LONG_QUOTIENT_THRESHOLD,
        NEWTON_RAPHSON_THRESHOLD,
        NEWTON_QUOTIENT_THRESHOLD,
    ]
    .into_iter()
    .flat_map(|cutoff| {
        // Four discarded guards require a positive quotient; widths below
        // twelve use the smallest valid constructed geometry instead.
        let boundary = cutoff.max(12);
        [
            (boundary - 1, 4),
            (boundary, 4),
            (boundary + 1, 4),
            (2 * boundary - 2, 0),
            (2 * boundary, 0),
            (2 * boundary + 2, 0),
        ]
    }) {
        for leading in [1, Limb::MAX] {
            let mut den = alloc::vec![Limb::MAX; width];
            *den.last_mut().expect("nonempty divisor") = leading;
            let divisor = InternalMpUint::from_limbs(den);
            let expected =
                InternalMpUint::from_limbs(alloc::vec![Limb::MAX; (width >> 1) - guards]);
            let product = divisor.mul(&expected);
            for (numerator, quotient) in [
                (
                    product.sub(&InternalMpUint::one()),
                    expected.sub(&InternalMpUint::one()),
                ),
                (product.clone(), expected.clone()),
                (
                    product.add(&divisor.sub(&InternalMpUint::one())),
                    expected.clone(),
                ),
            ] {
                let prior = output.clone();
                if !Division::truncated_quotient::<false>(&numerator, &divisor, &mut output) {
                    assert_eq!(output, prior);
                    Division::div_into::<false, false>(
                        &numerator,
                        &divisor,
                        &mut output,
                        &mut DivScratch::default(),
                    );
                }
                assert_eq!(output, quotient);
            }
            let prior = output.clone();
            if !Division::truncated_quotient::<true>(&product, &divisor, &mut output) {
                assert_eq!(output, prior);
                Division::div_into::<false, true>(
                    &product,
                    &divisor,
                    &mut output,
                    &mut DivScratch::default(),
                );
            }
            assert_eq!(output, expected);
        }
    }
}

#[test]
fn fixed_equal_width_and_exact_multiple_shapes_match_full_division() {
    let mut den = alloc::vec![0x9e37; 24];
    *den.last_mut().expect("nonempty divisor") = 3;
    let near_divisor = InternalMpUint::from_limbs(alloc::vec![0xdead; 30]);
    for (numerator, divisor) in [
        (
            InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 24]),
            InternalMpUint::from_limbs(den),
        ),
        (
            near_divisor.mul(&InternalMpUint::from_limb(0x0123)),
            near_divisor,
        ),
    ] {
        let mut quotient = InternalMpUint::zero();
        assert!(Division::truncated_quotient::<false>(
            &numerator,
            &divisor,
            &mut quotient
        ));
        assert_eq!(quotient, numerator.div_rem(&divisor).0);
    }
}
