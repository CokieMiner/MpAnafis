//! Newton quotients and remainders against Algorithm D, including the crossover.

extern crate std;

use core::cmp::Ordering;

use proptest::prelude::*;

use super::{DivScratch, Division, InternalMpUint, Limb, NEWTON_RAPHSON_THRESHOLD};

#[test]
fn truncated_prefix_requires_three_corrections_and_preserves_exact_output() {
    let mut found = false;
    let mut scratch = DivScratch::default();
    'search: for fraction in [129_usize, 130] {
        // D/B^2 is fraction/256, within the normalized interval [1/2,1).
        // All supported limb widths are at least 16 bits, so both shifts fit.
        let divisor = InternalMpUint::from_limbs(alloc::vec![0, fraction << (Limb::BITS - 8),]);
        let (reciprocal, discarded) =
            Division::newton_block_reciprocal::<false>(divisor.limbs(), 1, &mut scratch);
        let inverse = reciprocal
            .limbs()
            .get(discarded..)
            .expect("inverse high part");
        for offset in 0..128 {
            let high = (3_usize << (Limb::BITS - 3)) + offset;
            let numerator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX, Limb::MAX, high]);
            let mut quotient = InternalMpUint::zero();
            let mut remainder = InternalMpUint::zero();
            let _ = Division::algorithm_d::<true, true, false, false>(
                numerator.limbs(),
                divisor.limbs(),
                &mut quotient,
                &mut remainder,
                &mut scratch,
            );
            let _ = Division::newton_quotient_estimate(&[high], inverse, &mut scratch);
            let start = scratch
                .v_padded
                .len()
                .checked_sub(2)
                .expect("estimate and guard");
            let estimate = InternalMpUint::from_limbs_slice(
                scratch
                    .v_padded
                    .get(start..)
                    .expect("initialized estimate suffix"),
            );
            let error = quotient.sub(&estimate);
            assert!(error <= InternalMpUint::from_limb(3));
            if error != InternalMpUint::from_limb(3) {
                continue;
            }
            let mut actual = InternalMpUint::zero();
            let mut actual_remainder = InternalMpUint::zero();
            Division::newton::<true, true, false>(
                &numerator,
                &divisor,
                &mut actual,
                &mut actual_remainder,
                &mut scratch,
            );
            assert_eq!(actual, quotient);
            assert_eq!(actual_remainder, remainder);
            let exact = divisor.mul(&quotient);
            Division::newton::<true, false, true>(
                &exact,
                &divisor,
                &mut actual,
                &mut actual_remainder,
                &mut scratch,
            );
            assert_eq!(actual, quotient);
            assert!(actual_remainder.is_zero());
            found = true;
            break 'search;
        }
    }
    assert!(
        found,
        "the normalized family contains a three-correction estimate"
    );
}

proptest! {
    #[test]
    #[expect(
        unsafe_code,
        reason = "Constructed normalized divisors and guarded initialized dividend buffers satisfy the block division kernel contract."
    )]
    fn quotient_only_blocks_cover_zero_one_and_saturated_guards(
        mut denominator in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 9 } else { 140 }),
        mut digits in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 19 } else { 360 }),
        upper_guard in prop_oneof![Just(0), Just(1), Just(2), Just(Limb::MAX), any::<Limb>()],
    ) {
        *denominator.last_mut().expect("nonempty divisor") |= 1 << (Limb::BITS - 1);
        *digits.last_mut().expect("nonempty quotient") |= 1;
        *digits.get_mut(1).expect("quotient has at least two limbs") = upper_guard;
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(digits);
        let product = divisor.mul(&expected);
        let n = divisor.limbs().len();
        let mut scratch = DivScratch::default();
        for residue in [InternalMpUint::zero(), InternalMpUint::one(), divisor.sub(&InternalMpUint::one())] {
            let numerator = product.add(&residue);
            let mut input = numerator.limbs().to_vec();
            let width = input.len().max(n).checked_add(1).expect("bounded normalized width");
            input.resize(width, 0);
            let quotient_width = width.checked_sub(n).expect("dividend has a guard");
            let mut output = alloc::vec![Limb::MAX; quotient_width];
            // SAFETY: the divisor is normalized; one zero limb above U makes
            // its high divisor-width window smaller than D. Input has n plus
            // quotient_width initialized limbs, disjoint from output and scratch.
            unsafe {
                Division::newton_div_blocks::<true, false, false>(
                    &mut input, divisor.limbs(), output.as_mut_ptr(), quotient_width, &mut scratch,
                );
            }
            prop_assert_eq!(InternalMpUint::from_limbs(output), expected.clone());
        }
    }

    #[test]
    fn block_division_matches_constructed_values_and_reuses_scratch(
        mut denominator in proptest::collection::vec(any::<Limb>(), 2..=if cfg!(miri) { 9 } else { 140 }),
        quotient in proptest::collection::vec(prop_oneof![Just(0), Just(Limb::MAX), any::<Limb>()], 1..=if cfg!(miri) { 19 } else { 360 }),
        normalization in 0_u32..Limb::BITS,
    ) {
        let top = denominator.last_mut().expect("nonempty divisor");
        *top = (*top | (1 << Limb::BITS.wrapping_sub(1))) >> normalization;
        let divisor = InternalMpUint::from_limbs(denominator);
        let expected = InternalMpUint::from_limbs(quotient);
        let mut scratch = DivScratch::default();
        let mut actual = InternalMpUint::from_limb(Limb::MAX);
        let mut remainder = InternalMpUint::from_limb(Limb::MAX);
        for result in [InternalMpUint::zero(), divisor.sub(&InternalMpUint::one())] {
            let numerator = divisor.mul(&expected).add(&result);
            Division::newton::<true, true, false>(&numerator, &divisor, &mut actual, &mut remainder, &mut scratch);
            prop_assert_eq!(&actual, &expected);
            prop_assert_eq!(&remainder, &result);
            Division::newton::<true, false, false>(&numerator, &divisor, &mut actual, &mut remainder, &mut scratch);
            prop_assert_eq!(&actual, &expected);
            prop_assert!(remainder.is_zero());
            // A shorter exact quotient follows the same initialized scratch.
            Division::newton::<true, true, false>(&divisor, &divisor, &mut actual, &mut remainder, &mut scratch);
            prop_assert!(actual.is_one());
            prop_assert!(remainder.is_zero());
        }
    }

    #[test]
    fn prop_newton_terminates_for_normalized_denominators(
        n in prop_oneof![Just(1_usize), Just(2), Just(5), Just(10), Just(20), Just(40), Just(45), Just(60)],
        den_seed in any::<[Limb; 60]>(),
        num_seed in any::<[Limb; 120]>(),
    ) {
        let mut scratch = DivScratch::default();
        let mut den_limbs = den_seed
            .get(..n)
            .expect("n is bounded by the den_seed array length")
            .to_vec();
        if let Some(last_limb) = den_limbs.last_mut() {
            *last_limb |= 1 << (Limb::BITS - 1);
        }
        let den = InternalMpUint::from_limbs(den_limbs);

        let num_limbs = num_seed
            .get(..n.wrapping_mul(2))
            .expect("2n is bounded by the num_seed array length")
            .to_vec();
        let num = InternalMpUint::from_limbs(num_limbs);

        let mut q_newton = InternalMpUint::zero();
        let mut r_newton = InternalMpUint::zero();
        Division::newton::<true, true, false>(&num, &den, &mut q_newton, &mut r_newton, &mut scratch);

        let mut q_expected = InternalMpUint::zero();
        let mut r_expected = InternalMpUint::zero();
        if !Division::trivial::<true, true>(&num, &den, &mut q_expected, &mut r_expected) {
            let _ = Division::algorithm_d::<true, true, false, false>(
                num.limbs(), den.limbs(), &mut q_expected, &mut r_expected, &mut scratch,
            );
        }
        prop_assert_eq!(q_newton, q_expected, "quotient mismatch for n={}", n);
        prop_assert_eq!(r_newton, r_expected, "remainder mismatch for n={}", n);
    }

    /// The quotient-only policy skips the denormalizing pass and the remainder
    /// copy, so it must agree limb for limb with the full policy on the quotient.
    #[test]
    fn prop_newton_quotient_only_matches_full_policy(
        n in prop_oneof![Just(1_usize), Just(2), Just(5), Just(20), Just(45), Just(60)],
        den_seed in any::<[Limb; 60]>(),
        num_seed in any::<[Limb; 120]>(),
    ) {
        let mut scratch = DivScratch::default();
        let mut den_limbs = den_seed
            .get(..n)
            .expect("n is bounded by the den_seed array length")
            .to_vec();
        if let Some(last_limb) = den_limbs.last_mut() {
            *last_limb |= 1 << (Limb::BITS - 1);
        }
        let den = InternalMpUint::from_limbs(den_limbs);
        let num = InternalMpUint::from_limbs(
            num_seed
                .get(..n.wrapping_mul(2))
                .expect("2n is bounded by the num_seed array length")
                .to_vec(),
        );

        let mut q_full = InternalMpUint::zero();
        let mut r_full = InternalMpUint::zero();
        Division::newton::<true, true, false>(&num, &den, &mut q_full, &mut r_full, &mut scratch);
        let mut q_only = InternalMpUint::zero();
        let mut discarded = InternalMpUint::zero();
        Division::newton::<true, false, false>(&num, &den, &mut q_only, &mut discarded, &mut scratch);
        prop_assert_eq!(q_only, q_full, "quotient mismatch for n={}", n);
        let mut sentinel = InternalMpUint::from_limb(7);
        Division::newton::<false, true, false>(&num, &den, &mut sentinel, &mut discarded, &mut scratch);
        prop_assert_eq!(sentinel, InternalMpUint::from_limb(7));
        prop_assert_eq!(discarded, r_full, "remainder mismatch for n={}", n);
    }

    #[test]
    fn prop_newton_reciprocal_and_div_match_algorithm_d(
        n in prop_oneof![Just(1_usize), Just(2), Just(5), Just(10), Just(20), Just(45), Just(60)],
        den_seed in any::<[Limb; 60]>(),
        num_seed in any::<[Limb; 120]>(),
    ) {
        let mut scratch = DivScratch::default();
        let mut den_limbs = den_seed
            .get(..n)
            .expect("n is bounded by the den_seed array length")
            .to_vec();
        if let Some(last_limb) = den_limbs.last_mut() {
            *last_limb |= 1 << (Limb::BITS - 1);
        }
        let den = InternalMpUint::from_limbs(den_limbs);

        let num_limbs = num_seed
            .get(..n.wrapping_mul(2))
            .expect("2n is bounded by the num_seed array length")
            .to_vec();
        let num = InternalMpUint::from_limbs(num_limbs);

        let mut q_newton = InternalMpUint::zero();
        let mut r_newton = InternalMpUint::zero();
        Division::newton::<true, true, false>(&num, &den, &mut q_newton, &mut r_newton, &mut scratch);

        let mut q_expected = InternalMpUint::zero();
        let mut r_expected = InternalMpUint::zero();
        if !Division::trivial::<true, true>(&num, &den, &mut q_expected, &mut r_expected) {
            let _ = Division::algorithm_d::<true, true, false, false>(
                num.limbs(), den.limbs(), &mut q_expected, &mut r_expected, &mut scratch,
            );
        }

        prop_assert_eq!(q_newton, q_expected, "Newton quotient mismatch for n={}", n);
        prop_assert_eq!(r_newton, r_expected, "Newton remainder mismatch for n={}", n);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    #[cfg_attr(miri, ignore = "The production Newton crossover uses large native operands; bounded block-division properties run under Miri.")]
    fn prop_newton_threshold_division_identity(
        n in prop_oneof![Just(NEWTON_RAPHSON_THRESHOLD), Just(NEWTON_RAPHSON_THRESHOLD.wrapping_add(1))],
        den_seed in proptest::collection::vec(any::<Limb>(), NEWTON_RAPHSON_THRESHOLD.wrapping_add(1)),
        num_seed in proptest::collection::vec(
            any::<Limb>(),
            NEWTON_RAPHSON_THRESHOLD.wrapping_mul(2).wrapping_add(2),
        ),
    ) {
        let mut scratch = DivScratch::default();
        let mut den_limbs = den_seed
            .get(..n)
            .expect("n is bounded by the den_seed vector length")
            .to_vec();
        if let Some(last_limb) = den_limbs.last_mut() {
            *last_limb |= 1 << (Limb::BITS - 1);
        }
        let den = InternalMpUint::from_limbs(den_limbs);

        let num_limbs = num_seed
            .get(..n.wrapping_mul(2))
            .expect("2n is bounded by the num_seed vector length")
            .to_vec();
        let num = InternalMpUint::from_limbs(num_limbs);

        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        Division::newton::<true, true, false>(&num, &den, &mut quotient, &mut remainder, &mut scratch);
        prop_assert!(
            remainder.cmp(&den) == Ordering::Less,
            "remainder must be less than divisor for n={}",
            n,
        );
        let recombined = quotient.mul(&den).add(&remainder);
        prop_assert_eq!(recombined, num, "division identity failed for n={}", n);
    }
}

/// Exercises the exact division identity at the dispatch crossover and just
/// above it. The quotient shapes include a one-limb quotient, a balanced
/// quotient, exact division, and the largest valid remainder.
#[test]
#[cfg_attr(
    miri,
    ignore = "Production Newton crossover and long quotient shapes require native execution."
)]
fn newton_division_identity_at_production_sizes() {
    let above_threshold = NEWTON_RAPHSON_THRESHOLD
        .checked_add(1)
        .expect("production threshold fits");
    for n in [NEWTON_RAPHSON_THRESHOLD, above_threshold] {
        let high_bit = 1 << Limb::BITS.wrapping_sub(1);
        let denominator = InternalMpUint::from_limbs(
            (0..n)
                .map(|i| Limb::MAX.wrapping_sub(i.wrapping_mul(40_503)) | high_bit)
                .collect(),
        );
        let quotient_shapes = [
            InternalMpUint::from_limb(1),
            InternalMpUint::from_limbs(alloc::vec![Limb::MAX; n]),
            InternalMpUint::from_limbs(
                alloc::vec![Limb::MAX; n.checked_mul(3).expect("long quotient width")],
            ),
        ];
        for quotient in quotient_shapes {
            for remainder in [
                InternalMpUint::zero(),
                denominator.sub(&InternalMpUint::one()),
            ] {
                let numerator = denominator.mul(&quotient).add(&remainder);
                let mut scratch = DivScratch::default();
                let mut actual_quotient = InternalMpUint::zero();
                let mut actual_remainder = InternalMpUint::zero();
                Division::newton::<true, true, false>(
                    &numerator,
                    &denominator,
                    &mut actual_quotient,
                    &mut actual_remainder,
                    &mut scratch,
                );
                assert_eq!(actual_quotient, quotient, "quotient identity at n={n}");
                assert_eq!(actual_remainder, remainder, "remainder identity at n={n}");
                assert!(actual_remainder < denominator);
                Division::newton::<true, false, false>(
                    &numerator,
                    &denominator,
                    &mut actual_quotient,
                    &mut actual_remainder,
                    &mut scratch,
                );
                assert_eq!(
                    actual_quotient, quotient,
                    "guarded quotient identity at n={n}"
                );
            }
        }
    }
}

#[test]
fn identical_leading_windows_require_distinct_quotients() {
    for width in [3_usize, 4, 5, 64, 129] {
        let mut limbs = alloc::vec![0; width];
        *limbs.first_mut().expect("nonempty divisor") = 1;
        *limbs.last_mut().expect("nonempty divisor") = 1 << Limb::BITS.wrapping_sub(1);
        let divisor = InternalMpUint::from_limbs(limbs);
        let multiple = divisor.add(&divisor);
        let predecessor = multiple.sub(&InternalMpUint::one());
        let prefix_start = width.checked_sub(1).expect("positive width");
        assert_eq!(
            multiple.limbs().get(prefix_start..),
            predecessor.limbs().get(prefix_start..),
            "quotient estimation sees the same dividend prefix"
        );
        let mut scratch = DivScratch::default();
        let mut quotient = InternalMpUint::zero();
        let mut discarded = InternalMpUint::zero();
        for (numerator, expected) in [(&predecessor, 1), (&multiple, 2)] {
            Division::newton::<true, false, false>(
                numerator,
                &divisor,
                &mut quotient,
                &mut discarded,
                &mut scratch,
            );
            assert_eq!(quotient, InternalMpUint::from_limb(expected));
        }
    }
}

fn power_of_base(exponent: usize) -> InternalMpUint {
    let mut limbs = alloc::vec![0; exponent.checked_add(1).expect("power width")];
    *limbs.last_mut().expect("power has a high limb") = 1;
    InternalMpUint::from_limbs(limbs)
}

#[test]
fn newton_leading_digit_preserves_balanced_block_boundaries() {
    for width in [2_usize, 3, 40, 41, 79, 80, 129] {
        let mut den_limbs = alloc::vec![Limb::MAX; width];
        *den_limbs.last_mut().expect("nonempty divisor") = (Limb::MAX >> 1).wrapping_add(7);
        let denominator = InternalMpUint::from_limbs(den_limbs);
        let mut scratch = DivScratch::default();
        let mut actual = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        for leading in [0, 1] {
            let fill = if leading == 0 { Limb::MAX } else { 1 };
            let mut quotient = alloc::vec![fill; width];
            quotient.push(leading);
            let expected = InternalMpUint::from_limbs(quotient);
            let residue = denominator.sub(&InternalMpUint::one());
            let numerator = denominator.mul(&expected).add(&residue);
            Division::newton::<true, true, false>(
                &numerator,
                &denominator,
                &mut actual,
                &mut remainder,
                &mut scratch,
            );
            assert_eq!(
                actual, expected,
                "quotient width {width}, leading {leading}"
            );
            assert_eq!(
                remainder, residue,
                "remainder width {width}, leading {leading}"
            );
        }
    }
}

#[test]
#[cfg(not(target_pointer_width = "16"))]
#[cfg_attr(
    miri,
    ignore = "Mersenne division at 3072 to 8193 limbs exercises native transform crossovers; bounded residue and block properties run under Miri"
)]
fn newton_mersenne_remainders_cover_exact_and_near_multiples() {
    for width in [3072, 3073, 4095, 4096, 4097, 8191, 8192, 8193] {
        let denominator = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let mut scratch = DivScratch::default();
        let mut actual = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        for quotient_width in [width.div_euclid(2), width] {
            let expected = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; quotient_width]);
            // (B^n-1)(B^k-1) = B^(n+k)-B^n-B^k+1, constructed without
            // multiplication so the oracle does not share the transform path.
            let mut product_limbs = alloc::vec![Limb::MAX; width.checked_add(quotient_width).expect("test product width")];
            product_limbs
                .get_mut(..quotient_width)
                .expect("low block exists")
                .fill(0);
            *product_limbs.first_mut().expect("nonempty product") = 1;
            *product_limbs.get_mut(width).expect("high block exists") = Limb::MAX.wrapping_sub(1);
            let product = InternalMpUint::from_limbs(product_limbs);
            for residue in [
                InternalMpUint::zero(),
                InternalMpUint::one(),
                InternalMpUint::from_limb(2),
                InternalMpUint::from_limb(3),
                denominator.sub(&InternalMpUint::one()),
            ] {
                let numerator = product.add(&residue);
                Division::newton::<true, true, false>(
                    &numerator,
                    &denominator,
                    &mut actual,
                    &mut remainder,
                    &mut scratch,
                );
                assert_eq!(
                    actual, expected,
                    "quotient at {width}/{quotient_width} limbs"
                );
                assert_eq!(
                    remainder, residue,
                    "remainder at {width}/{quotient_width} limbs"
                );
            }
        }
    }
}

#[test]
fn newton_quotient_blocks_initialize_reused_zero_digits() {
    for width in [2_usize, 4, 5, 41, 129] {
        let divisor = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]);
        let mut scratch = DivScratch::default();
        let mut actual = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 1024]);
        let mut remainder = actual.clone();
        // Powers of the limb base make entire quotient blocks zero. A later
        // shorter quotient must overwrite both the old digits and its guard.
        for exponent in [width.checked_mul(3).expect("small test width"), width, 1, 0] {
            let quotient = power_of_base(exponent);
            let residue = divisor.sub(&InternalMpUint::one());
            let dividend = divisor.mul(&quotient).add(&residue);
            Division::newton::<true, true, false>(
                &dividend,
                &divisor,
                &mut actual,
                &mut remainder,
                &mut scratch,
            );
            assert_eq!(actual, quotient);
            assert_eq!(remainder, residue);
            actual = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 1024]);
            Division::newton::<true, false, false>(
                &dividend,
                &divisor,
                &mut actual,
                &mut remainder,
                &mut scratch,
            );
            assert_eq!(actual, quotient);
            Division::newton::<false, true, false>(
                &dividend,
                &divisor,
                &mut actual,
                &mut remainder,
                &mut scratch,
            );
            assert_eq!(
                actual, quotient,
                "remainder-only division leaves quotient output untouched"
            );
            assert_eq!(remainder, residue);
        }
    }
}
