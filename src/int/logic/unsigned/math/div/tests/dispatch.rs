//! Division tower identities, output modes, and recursive block shapes.

use core::cmp::Ordering;

use proptest::prelude::*;

use super::{DivScratch, Division, InternalMpUint, Limb};

fn dense_shape(limbs: usize, top: Limb) -> InternalMpUint {
    let mut values = alloc::vec![Limb::MAX; limbs];
    *values.last_mut().expect("test shape must have one limb") = top;
    InternalMpUint::from_limbs(values)
}

fn joined_blocks(low: &[Limb], high: &[Limb]) -> InternalMpUint {
    let mut limbs = low.to_vec();
    limbs.extend_from_slice(high);
    InternalMpUint::from_limbs(limbs)
}

fn burnikel_matches_algorithm_d(
    numerator: &InternalMpUint,
    denominator: &InternalMpUint,
    scratch: &mut DivScratch,
) -> (InternalMpUint, InternalMpUint) {
    let mut actual_quotient = InternalMpUint::zero();
    let mut actual_remainder = InternalMpUint::zero();
    Division::burnikel_ziegler::<true>(
        numerator,
        denominator,
        &mut actual_quotient,
        &mut actual_remainder,
        scratch,
    );

    let mut expected_quotient = InternalMpUint::zero();
    let mut expected_remainder = InternalMpUint::zero();
    let mut expected_scratch = DivScratch::default();
    let _ = Division::algorithm_d::<true, true, false, false>(
        numerator.limbs(),
        denominator.limbs(),
        &mut expected_quotient,
        &mut expected_remainder,
        &mut expected_scratch,
    );

    assert_eq!(actual_quotient, expected_quotient);
    assert_eq!(actual_remainder, expected_remainder);
    assert_eq!(
        actual_quotient.mul(denominator).add(&actual_remainder),
        *numerator,
        "division identity"
    );
    assert_eq!(actual_remainder.cmp(denominator), Ordering::Less);
    (actual_quotient, actual_remainder)
}

/// Pairs across the recursion's split width: every divisor width to 320
/// limbs, odd and even, with quotients from one limb to past two divisor
/// widths, so every leading block size occurs. Arbitrary nonzero top limbs
/// cover every normalization shift; saturated numerators drive the block
/// repairs toward their add-back bound.
fn recursive_width_pair() -> impl Strategy<Value = (InternalMpUint, InternalMpUint)> {
    (
        2_usize..=if cfg!(miri) { 9 } else { 320 },
        1_usize..=if cfg!(miri) { 10 } else { 700 },
        any::<bool>(),
    )
        .prop_flat_map(|(width, quotient, saturated)| {
            (
                proptest::collection::vec(any::<Limb>(), width),
                proptest::collection::vec(any::<Limb>(), width.wrapping_add(quotient)),
                Just(saturated),
            )
        })
        .prop_map(|(mut divisor, mut numerator, saturated)| {
            if let Some(top) = divisor.last_mut() {
                *top |= 1;
            }
            if saturated {
                numerator.fill(Limb::MAX);
            }
            (
                InternalMpUint::from_limbs(numerator),
                InternalMpUint::from_limbs(divisor),
            )
        })
}

/// Normalized 2n/n and 3n/2n windows with both leading quotient regimes.
fn burnikel_shape_pair() -> impl Strategy<Value = (InternalMpUint, InternalMpUint)> {
    let top_bit = (Limb::MAX >> 1).wrapping_add(1);
    let random_pair = (2_usize..=24).prop_flat_map(move |n| {
        let den = dense_shape(n, top_bit.wrapping_add(1));
        let numerator_lengths = prop_oneof![
            Just(n.wrapping_mul(2)),
            Just(n.wrapping_add(n.wrapping_div(2)))
        ];
        numerator_lengths.prop_map(move |num_len| (dense_shape(num_len, top_bit), den.clone()))
    });
    prop_oneof![
        Just((
            dense_shape(512, top_bit),
            dense_shape(256, top_bit.wrapping_add(8)),
        )),
        Just((
            dense_shape(512, Limb::MAX),
            dense_shape(256, top_bit.wrapping_add(8)),
        )),
        Just((
            dense_shape(384, top_bit),
            dense_shape(256, top_bit.wrapping_add(8)),
        )),
        Just((
            dense_shape(384, Limb::MAX),
            dense_shape(256, top_bit.wrapping_add(8)),
        )),
        random_pair,
    ]
}

#[test]
fn burnikel_direct_paths_handle_nonzero_normalization_shift() {
    let denominator = dense_shape(8, Limb::MAX >> 1);
    assert_eq!(
        denominator
            .limbs()
            .last()
            .expect("test divisor is nonzero")
            .leading_zeros(),
        1
    );

    for numerator in [
        dense_shape(16, Limb::MAX >> 2),
        dense_shape(12, Limb::MAX >> 2),
    ] {
        let mut scratch = DivScratch::default();
        drop(burnikel_matches_algorithm_d(
            &numerator,
            &denominator,
            &mut scratch,
        ));
    }
}

#[test]
fn burnikel_three_by_two_checks_exact_quotient_width_boundaries() {
    let top_bit = (Limb::MAX >> 1).wrapping_add(1);
    let mut denominator_limbs = alloc::vec![17; 8];
    *denominator_limbs
        .last_mut()
        .expect("test divisor is nonzero") = top_bit.wrapping_add(7);
    let denominator = InternalMpUint::from_limbs(denominator_limbs.clone());
    let low_block = alloc::vec![23; 4];

    // `a21 = V - 1` is the largest numerator prefix whose quotient still fits
    // in four limbs. Decrementing the low limb leaves the top divisor block
    // equal to the top window, the all-ones quotient estimate.
    let mut below_limbs = denominator_limbs.clone();
    let below_low = below_limbs.first_mut().expect("test divisor is nonzero");
    *below_low = below_low.wrapping_sub(1);
    let direct_numerator = joined_blocks(&low_block, &below_limbs);
    let mut direct_scratch = DivScratch::default();
    let (direct_quotient, _) =
        burnikel_matches_algorithm_d(&direct_numerator, &denominator, &mut direct_scratch);
    assert_eq!(direct_quotient.limbs(), &[Limb::MAX; 4]);

    // `a21 = V` makes the exact quotient B^4, a set high bit in the leading
    // block above the four-limb recursive quotient.
    let leading_numerator = joined_blocks(&low_block, &denominator_limbs);
    let mut leading_scratch = DivScratch::default();
    let (leading_quotient, leading_remainder) =
        burnikel_matches_algorithm_d(&leading_numerator, &denominator, &mut leading_scratch);
    assert_eq!(leading_quotient.limbs(), &[0, 0, 0, 0, 1]);
    assert_eq!(leading_remainder, InternalMpUint::from_limbs(low_block));
}

#[test]
fn burnikel_three_by_two_retains_leading_digit_through_recursive_corrections() {
    for width in [2_usize, 8, 64, 96, 128, 256] {
        let denominator = dense_shape(width, (Limb::MAX >> 1).wrapping_add(7));
        let low = alloc::vec![Limb::MAX; width.div_euclid(2)];
        let mut scratch = DivScratch::default();
        for high in [
            denominator.sub(&InternalMpUint::one()),
            denominator.clone(),
            denominator.add(&InternalMpUint::one()),
            InternalMpUint::from_limbs(alloc::vec![Limb::MAX; width]),
        ] {
            let numerator = joined_blocks(&low, high.limbs());
            drop(burnikel_matches_algorithm_d(
                &numerator,
                &denominator,
                &mut scratch,
            ));
        }
    }
}

#[test]
fn burnikel_two_by_one_removes_an_equal_upper_block() {
    let top_bit = (Limb::MAX >> 1).wrapping_add(1);
    let mut denominator_limbs = alloc::vec![29; 8];
    *denominator_limbs
        .last_mut()
        .expect("test divisor is nonzero") = top_bit.wrapping_add(11);
    let denominator = InternalMpUint::from_limbs(denominator_limbs.clone());
    let low_block = alloc::vec![31; 8];
    let numerator = joined_blocks(&low_block, &denominator_limbs);

    let mut scratch = DivScratch::default();
    let (quotient, remainder) =
        burnikel_matches_algorithm_d(&numerator, &denominator, &mut scratch);
    assert_eq!(quotient.limbs(), &[0, 0, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(remainder, InternalMpUint::from_limbs(low_block));
}

#[test]
fn burnikel_reuses_scratch_across_direct_miss_and_general_paths() {
    let top_bit = (Limb::MAX >> 1).wrapping_add(1);
    let denominator = dense_shape(8, top_bit.wrapping_add(8));
    let cases = [
        dense_shape(12, top_bit),
        dense_shape(12, Limb::MAX),
        dense_shape(16, Limb::MAX),
        dense_shape(17, top_bit),
    ];
    let mut scratch = DivScratch::default();

    for numerator in cases {
        drop(burnikel_matches_algorithm_d(
            &numerator,
            &denominator,
            &mut scratch,
        ));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]
    /// The unpadded recursion agrees with Algorithm D for every divisor
    /// width around its split size, including odd widths and all leading
    /// block sizes.
    #[test]
    fn prop_burnikel_matches_algorithm_d_across_split_widths(pair in recursive_width_pair()) {
        let (numerator, denominator) = pair;
        drop(burnikel_matches_algorithm_d(
            &numerator,
            &denominator,
            &mut DivScratch::default(),
        ));
    }

    /// Burnikel-Ziegler must agree with Algorithm D limb for limb on every
    /// shape its direct shortcuts and its general driver consume, and the
    /// algebraic identity `q * d + r == n` with `r < d` must hold throughout.
    #[test]
    #[cfg_attr(miri, ignore = "Fixed 256-limb Burnikel block shapes require native execution; bounded split-width properties run under Miri.")]
    fn prop_burnikel_matches_algorithm_d_on_direct_and_general_shapes(
        pair in burnikel_shape_pair(),
    ) {
        let (numerator, denominator) = pair;
        let mut actual_quotient = InternalMpUint::zero();
        let mut actual_remainder = InternalMpUint::zero();
        let mut actual_scratch = DivScratch::default();
        Division::burnikel_ziegler::<true>(
            &numerator,
            &denominator,
            &mut actual_quotient,
            &mut actual_remainder,
            &mut actual_scratch,
        );

        let mut expected_quotient = InternalMpUint::zero();
        let mut expected_remainder = InternalMpUint::zero();
        let mut expected_scratch = DivScratch::default();
        let _ = Division::algorithm_d::<true, true, false, false>(
            numerator.limbs(),
            denominator.limbs(),
            &mut expected_quotient,
            &mut expected_remainder,
            &mut expected_scratch,
        );

        prop_assert_eq!(&actual_quotient, &expected_quotient);
        prop_assert_eq!(&actual_remainder, &expected_remainder);
        let recombined = actual_quotient.mul(&denominator).add(&actual_remainder);
        prop_assert_eq!(&recombined, &numerator, "division identity");
        prop_assert!(
            actual_remainder.cmp(&denominator) == Ordering::Less,
            "remainder must be smaller than the divisor"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]
    #[test]
    fn output_modes_and_assignment_preserve_exact_division_identities(
        numerator_words in proptest::collection::vec(any::<Limb>(), 0..=if cfg!(miri) { 5 } else { 10 }),
        divisor_words in proptest::collection::vec(any::<Limb>(), 1..=if cfg!(miri) { 5 } else { 10 })
            .prop_filter("nonzero divisor", |words| words.iter().any(|&word| word != 0)),
        divisor_limb in 1_usize..=Limb::MAX,
    ) {
        let numerator = InternalMpUint::from_limbs(numerator_words);
        let wide_divisor = InternalMpUint::from_limbs(divisor_words);
        let scalar_divisor = InternalMpUint::from_limb(divisor_limb);
        let mut scratch = DivScratch::default();
        for denominator in [&wide_divisor, &scalar_divisor] {
            let (quotient, remainder) = numerator.div_rem(denominator);
            prop_assert_eq!(&quotient.mul(denominator).add(&remainder), &numerator);
            prop_assert!(remainder < *denominator);
            prop_assert_eq!(&numerator.div(denominator), &quotient);
            prop_assert_eq!(&numerator.rem(denominator), &remainder);
            let mut quotient_output = InternalMpUint::from_limbs(alloc::vec![Limb::MAX; 16]);
            let mut remainder_output = quotient_output.clone();
            Division::div_rem_into(&numerator, denominator, &mut quotient_output, &mut remainder_output, &mut scratch);
            prop_assert_eq!(&quotient_output, &quotient);
            prop_assert_eq!(&remainder_output, &remainder);
            remainder_output.set_limb(Limb::MAX);
            Division::rem_into(&numerator, denominator, &mut remainder_output, &mut scratch);
            prop_assert_eq!(&remainder_output, &remainder);
            let mut assigned_quotient = numerator.clone();
            let mut assigned_remainder = numerator.clone();
            assigned_quotient.div_assign(denominator);
            assigned_remainder.rem_assign(denominator);
            prop_assert_eq!(assigned_quotient, quotient);
            prop_assert_eq!(assigned_remainder, remainder);
        }
    }
}
