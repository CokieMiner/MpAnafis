//! Cross-entry-point properties for the division tower.
//!
//! Each property pins one entry point against another that is known to agree
//! with it, so a divergence in any single divider surfaces regardless of which
//! crossover the operands land on.

use core::cmp::Ordering;

use proptest::prelude::*;

use super::{BURNIKEL_ZIEGLER_THRESHOLD, DivScratch, Division, InternalMpUint, LIMB_BITS, Limb};

fn internal_uint(max_limbs: usize) -> impl Strategy<Value = InternalMpUint> {
    proptest::collection::vec(any::<usize>(), 0..=max_limbs).prop_map(InternalMpUint::from_limbs)
}

fn internal_uint_nonzero(max_limbs: usize) -> impl Strategy<Value = InternalMpUint> {
    internal_uint(max_limbs).prop_filter("denominator must be non-zero", |value| !value.is_zero())
}

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
    Division::burnikel_ziegler(
        numerator,
        denominator,
        &mut actual_quotient,
        &mut actual_remainder,
        scratch,
    );

    let mut expected_quotient = InternalMpUint::zero();
    let mut expected_remainder = InternalMpUint::zero();
    let mut expected_scratch = DivScratch::default();
    Division::algorithm_d(
        numerator,
        denominator,
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

/// Pairs shaped to hit the Burnikel-Ziegler direct paths and the general
/// block driver, with the minimized regression shapes pinned as fixed cases.
///
/// The four fixed cases are the exact shapes the direct-recursion shortcuts
/// were built for: normalized `2n / n` with both possible leading quotient
/// limbs, and normalized `3n / 2n` with the quotient below and above
/// `B^(n/2)` — the latter must fall back to the general driver instead of
/// overflowing the shortcut's quotient buffer. Random pairs vary the divisor
/// limb count and numerator-to-divisor width ratio with an already-normalized
/// divisor; focused tests below cover nonzero shifts and exact guard boundaries.
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

fn assert_divisibility_matches_remainder(
    value: &InternalMpUint,
    divisor: &InternalMpUint,
    expected: bool,
) {
    assert!(
        !divisor.is_zero(),
        "the remainder reference needs a divisor"
    );
    assert_eq!(value.is_divisible_by(divisor), expected);
    assert_eq!(value.rem(divisor).is_zero(), expected);
}

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
    let mut divisor_limbs = alloc::vec![3; 6];
    *divisor_limbs
        .last_mut()
        .expect("the test divisor is non-empty") = 1;
    let divisor = InternalMpUint::from_limbs(divisor_limbs);
    let exact = divisor.mul(&InternalMpUint::from_limb(7));
    assert_eq!(exact.limbs().len(), divisor.limbs().len());

    let one = InternalMpUint::one();
    assert_divisibility_matches_remainder(&exact.sub(&one), &divisor, false);
    assert_divisibility_matches_remainder(&exact, &divisor, true);
    assert_divisibility_matches_remainder(&exact.add(&one), &divisor, false);
}

#[test]
fn divisibility_removes_whole_and_partial_limb_power_of_two_factors() {
    let odd_divisor = InternalMpUint::from_limbs(alloc::vec![3, 5, 1]);
    let tz = LIMB_BITS
        .checked_add(3)
        .expect("one limb plus three bits fits usize");
    let divisor = odd_divisor.shl(tz);
    let valuation_unit = InternalMpUint::power_of_two(tz);

    // A one-limb factor keeps the operands equal-width and exercises the
    // shifted low-limb reconstruction in the scalar divisibility path.
    let equal_exact = divisor.mul(&InternalMpUint::from_limb(3));
    assert_eq!(equal_exact.limbs().len(), divisor.limbs().len());
    assert_divisibility_matches_remainder(&equal_exact, &divisor, true);
    assert_divisibility_matches_remainder(&equal_exact.add(&valuation_unit), &divisor, false);

    // The sparse three-limb factor forces the low-to-high exact-division loop.
    // Adding or subtracting `2^tz` preserves the necessary 2-adic condition, so
    // the false cases cannot be rejected by the valuation precheck alone.
    let factor = InternalMpUint::from_limbs(alloc::vec![5, 0, 1]);
    let unequal_exact = divisor.mul(&factor);
    assert!(unequal_exact.limbs().len() > divisor.limbs().len());
    assert_divisibility_matches_remainder(&unequal_exact, &divisor, true);
    assert_divisibility_matches_remainder(&unequal_exact.sub(&valuation_unit), &divisor, false);
    assert_divisibility_matches_remainder(&unequal_exact.add(&valuation_unit), &divisor, false);
}

#[test]
fn divisibility_matches_at_subquadratic_fallback_threshold_neighbors() {
    const {
        assert!(
            BURNIKEL_ZIEGLER_THRESHOLD >= 2,
            "the recursive division threshold must admit a multi-limb divisor"
        );
    }
    let mut divisor_limbs = alloc::vec![0; BURNIKEL_ZIEGLER_THRESHOLD];
    *divisor_limbs
        .first_mut()
        .expect("the threshold divisor is non-empty") = 3;
    *divisor_limbs
        .last_mut()
        .expect("the threshold divisor is non-empty") = 1;
    let divisor = InternalMpUint::from_limbs(divisor_limbs);
    let below = BURNIKEL_ZIEGLER_THRESHOLD
        .checked_sub(1)
        .expect("the threshold is positive");

    for quotient_shift in [below, BURNIKEL_ZIEGLER_THRESHOLD] {
        let shift = quotient_shift
            .checked_mul(LIMB_BITS)
            .expect("the test shift fits usize");
        let exact = divisor.shl(shift);
        assert_eq!(
            exact.limbs().len().checked_sub(divisor.limbs().len()),
            Some(quotient_shift),
        );
        assert_divisibility_matches_remainder(&exact, &divisor, true);

        let mut near = exact;
        near.increment();
        assert_divisibility_matches_remainder(&near, &divisor, false);
    }
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
        assert!(
            scratch.bz_a_pad.is_empty(),
            "exact normalized shape must bypass the general block driver"
        );
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
    // in four limbs. Decrementing the low limb leaves `a2 == b1`, so this also
    // exercises the special all-ones quotient estimate in `bz_div_3n2n`.
    let mut below_limbs = denominator_limbs.clone();
    let below_low = below_limbs.first_mut().expect("test divisor is nonzero");
    *below_low = below_low.wrapping_sub(1);
    let direct_numerator = joined_blocks(&low_block, &below_limbs);
    let mut direct_scratch = DivScratch::default();
    let (direct_quotient, _) =
        burnikel_matches_algorithm_d(&direct_numerator, &denominator, &mut direct_scratch);
    assert_eq!(direct_quotient.limbs(), &[Limb::MAX; 4]);
    assert!(
        direct_scratch.bz_a_pad.is_empty(),
        "a21 = V - 1 must take the direct 3n/2n path"
    );

    // `a21 = V` makes the exact quotient `B^4`, which needs a fifth limb and
    // therefore must be handled by the general block driver.
    let fallback_numerator = joined_blocks(&low_block, &denominator_limbs);
    let mut fallback_scratch = DivScratch::default();
    let (fallback_quotient, fallback_remainder) =
        burnikel_matches_algorithm_d(&fallback_numerator, &denominator, &mut fallback_scratch);
    assert_eq!(fallback_quotient.limbs(), &[0, 0, 0, 0, 1]);
    assert_eq!(fallback_remainder, InternalMpUint::from_limbs(low_block));
    assert!(
        !fallback_scratch.bz_a_pad.is_empty(),
        "a21 = V must fall back instead of overflowing the direct quotient"
    );
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
    assert!(
        scratch.bz_a_pad.is_empty(),
        "upper = V must take the direct 2n/n path after one subtraction"
    );
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

#[test]
fn algorithm_d_normalization_boundary_covers_output_modes() {
    for top in [Limb::MAX, Limb::MAX >> 1] {
        let denominator = InternalMpUint::from_limbs(alloc::vec![5, top]);
        let stack_numerator = dense_shape(63, Limb::MAX);
        let mut stack_scratch = DivScratch::default();
        let mut expected_quotient = InternalMpUint::zero();
        let mut expected_remainder = InternalMpUint::zero();
        Division::algorithm_d(
            &stack_numerator,
            &denominator,
            &mut expected_quotient,
            &mut expected_remainder,
            &mut stack_scratch,
        );
        assert!(stack_scratch.u_norm.is_empty());
        assert!(stack_scratch.v_norm.is_empty());
        assert_eq!(
            expected_quotient.mul(&denominator).add(&expected_remainder),
            stack_numerator
        );
        assert!(expected_remainder < denominator);

        let untouched_remainder = InternalMpUint::from_limb(37);
        let mut quotient_only = InternalMpUint::zero();
        let mut remainder_sentinel = untouched_remainder.clone();
        assert!(Division::try_algorithm_d_unscratched::<true, false, false>(
            &stack_numerator,
            &denominator,
            &mut quotient_only,
            &mut remainder_sentinel,
        ));
        assert_eq!(quotient_only, expected_quotient);
        assert_eq!(remainder_sentinel, untouched_remainder);

        let untouched_quotient = InternalMpUint::from_limb(41);
        let mut quotient_sentinel = untouched_quotient.clone();
        let mut remainder_only = InternalMpUint::zero();
        assert!(Division::try_algorithm_d_unscratched::<false, true, true>(
            &stack_numerator,
            &denominator,
            &mut quotient_sentinel,
            &mut remainder_only,
        ));
        assert_eq!(quotient_sentinel, untouched_quotient);
        assert_eq!(remainder_only, expected_remainder);

        let heap_numerator = dense_shape(64, Limb::MAX);
        let mut heap_scratch = DivScratch::default();
        let mut heap_quotient = InternalMpUint::zero();
        let mut heap_remainder = InternalMpUint::zero();
        Division::algorithm_d(
            &heap_numerator,
            &denominator,
            &mut heap_quotient,
            &mut heap_remainder,
            &mut heap_scratch,
        );
        assert_eq!(heap_scratch.u_norm.len(), 65);
        assert_eq!(heap_scratch.v_norm.len(), 2);
        assert_eq!(
            heap_quotient.mul(&denominator).add(&heap_remainder),
            heap_numerator
        );
        assert!(heap_remainder < denominator);

        let mut heap_remainder_only = InternalMpUint::zero();
        Division::algorithm_d_rem(
            &heap_numerator,
            &denominator,
            &mut heap_remainder_only,
            &mut heap_scratch,
        );
        assert_eq!(heap_remainder_only, heap_remainder);

        let untouched_heap_quotient = InternalMpUint::from_limb(43);
        let untouched_heap_remainder = InternalMpUint::from_limb(47);
        let mut heap_quotient_sentinel = untouched_heap_quotient.clone();
        let mut heap_remainder_sentinel = untouched_heap_remainder.clone();
        assert!(
            !Division::try_algorithm_d_unscratched::<true, false, false>(
                &heap_numerator,
                &denominator,
                &mut heap_quotient_sentinel,
                &mut heap_remainder_sentinel,
            )
        );
        assert_eq!(heap_quotient_sentinel, untouched_heap_quotient);
        assert_eq!(heap_remainder_sentinel, untouched_heap_remainder);
    }
}

proptest! {
    /// Burnikel-Ziegler must agree with Algorithm D limb for limb on every
    /// shape its direct shortcuts and its general driver consume, and the
    /// algebraic identity `q * d + r == n` with `r < d` must hold throughout.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest body operates on bounded shape sizes only"
    )]
    #[test]
    fn prop_burnikel_matches_algorithm_d_on_direct_and_general_shapes(
        pair in burnikel_shape_pair(),
    ) {
        let (numerator, denominator) = pair;
        let mut actual_quotient = InternalMpUint::zero();
        let mut actual_remainder = InternalMpUint::zero();
        let mut actual_scratch = DivScratch::default();
        Division::burnikel_ziegler(
            &numerator,
            &denominator,
            &mut actual_quotient,
            &mut actual_remainder,
            &mut actual_scratch,
        );

        let mut expected_quotient = InternalMpUint::zero();
        let mut expected_remainder = InternalMpUint::zero();
        let mut expected_scratch = DivScratch::default();
        Division::algorithm_d(
            &numerator,
            &denominator,
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
    #[test]
    fn prop_div_rem_into_matches_the_value_entry_point(
        numerator in internal_uint(10),
        denominator in internal_uint_nonzero(6),
    ) {
        let mut quotient_out = InternalMpUint::zero();
        let mut remainder_out = InternalMpUint::zero();
        let mut scratch = DivScratch::default();

        Division::div_rem_into(
            &numerator,
            &denominator,
            &mut quotient_out,
            &mut remainder_out,
            &mut scratch,
        );
        let (expected_quotient, expected_remainder) =
            numerator.div_rem(&denominator);
        prop_assert_eq!(quotient_out.limbs(), expected_quotient.limbs());
        prop_assert_eq!(remainder_out.limbs(), expected_remainder.limbs());
    }

    #[test]
    fn prop_rem_into_matches_the_value_entry_point(
        numerator in internal_uint(10),
        denominator in internal_uint_nonzero(10),
    ) {
        let mut remainder_out = InternalMpUint::zero();
        let mut scratch = DivScratch::default();

        Division::rem_into(
            &numerator,
            &denominator,
            &mut remainder_out,
            &mut scratch,
        );

        let expected_remainder = numerator.rem(&denominator);
        prop_assert_eq!(remainder_out.limbs(), expected_remainder.limbs());
        prop_assert!(remainder_out.cmp(&denominator) == Ordering::Less);
    }

    #[test]
    fn prop_div_rem_satisfies_quotient_remainder_identity(
        numerator in internal_uint(10),
        denominator in internal_uint_nonzero(6),
    ) {
        let (quotient, remainder) = numerator.div_rem(&denominator);
        let recombined = quotient.mul(&denominator).add(&remainder);
        prop_assert_eq!(recombined.limbs(), numerator.limbs());
        prop_assert!(remainder.cmp(&denominator) == Ordering::Less || remainder.is_zero());
    }

    /// The quotient-only path is the one the truncation shortcut lives on, so it
    /// gets pinned against `div_rem` on every shape the strategies produce.
    #[test]
    fn prop_div_and_rem_agree_with_div_rem(
        numerator in internal_uint(10),
        denominator in internal_uint_nonzero(6),
    ) {
        let (expected_quotient, expected_remainder) = numerator.div_rem(&denominator);
        let quotient = numerator.div(&denominator);
        let remainder = numerator.rem(&denominator);

        prop_assert_eq!(quotient.limbs(), expected_quotient.limbs());
        prop_assert_eq!(remainder.limbs(), expected_remainder.limbs());
    }

    #[test]
    fn prop_assign_forms_match_the_value_forms(
        numerator in internal_uint(10),
        denominator in internal_uint_nonzero(6),
    ) {
        let expected_quotient = numerator.div(&denominator);
        let expected_remainder = numerator.rem(&denominator);

        let mut quotient_value = numerator.clone();
        let mut remainder_value = numerator;
        quotient_value.div_assign(&denominator);
        remainder_value.rem_assign(&denominator);

        prop_assert_eq!(quotient_value.limbs(), expected_quotient.limbs());
        prop_assert_eq!(remainder_value.limbs(), expected_remainder.limbs());
    }

    #[test]
    fn prop_single_limb_division_recombines(
        numerator in internal_uint(8),
        denominator_limb in any::<usize>()
            .prop_filter("denominator must be non-zero", |value| *value != 0),
    ) {
        let denominator = InternalMpUint::from_limb(denominator_limb);
        let quotient = numerator.div(&denominator);
        let remainder = numerator.rem(&denominator);
        let recombined = quotient.mul(&denominator).add(&remainder);
        prop_assert_eq!(recombined.limbs(), numerator.limbs());
        prop_assert!(remainder.cmp(&denominator) == Ordering::Less || remainder.is_zero());
    }
}
