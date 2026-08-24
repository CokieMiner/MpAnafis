//! Quotient shortcuts must agree with forced Algorithm D, always.
//!
//! Truncation is only sound because the error is provably in `{0, 1}` and the
//! `r' >= q'` test decides which. These properties exercise both outcomes:
//! random operands take the cheap branch almost every time, while exact
//! multiples take the correcting branch every time.

use alloc::vec::Vec;

use proptest::prelude::*;

use crate::int::types::INLINE_LIMBS;

use super::{DivScratch, Division, InternalMpUint, Limb};

/// Builds an operand whose top limb is non-zero, so the limb count is exact.
fn dense_operand(limbs: &[Limb]) -> InternalMpUint {
    let mut owned: Vec<Limb> = limbs.to_vec();
    if let Some(top) = owned.last_mut()
        && *top == 0
    {
        *top = 1;
    }
    InternalMpUint::from_limbs(owned)
}

/// Equal-width operand pairs spanning every outcome of the leading-limb
/// comparison, with the minimized regression shapes pinned as fixed cases.
///
/// The fixed all-ones dividend against the fixed 24-limb divisor
/// `[0x9e37; 24]` topped by 3 is the original static regression: equal-width
/// operands whose quotient is far above the zero-or-one comparison, which
/// must fall through to the truncated machinery.  Random pairs build the
/// dividend as `k * den + r` for a small scalar `k`, which lands the quotient
/// in the `0`, `1`, and small-scalar regions.
fn equal_width_pair() -> impl Strategy<Value = (InternalMpUint, InternalMpUint)> {
    let mut fixed_den: Vec<Limb> = alloc::vec![0x9e37; 24];
    if let Some(last) = fixed_den.last_mut() {
        *last = 3;
    }
    let random_pair = (
        proptest::collection::vec(any::<Limb>(), 2..=40),
        any::<u8>().prop_map(Limb::from),
        proptest::collection::vec(any::<Limb>(), 0..=1),
    )
        .prop_map(|(den_limbs, multiple, tail_limbs)| {
            let den_b = dense_operand(&den_limbs);
            let scaled = den_b.mul(&InternalMpUint::from_limbs(alloc::vec![multiple]));
            let num_a = scaled.add(&dense_operand(&tail_limbs));
            (num_a, den_b)
        });
    prop_oneof![
        Just((
            dense_operand(&alloc::vec![Limb::MAX; 24]),
            dense_operand(&fixed_den),
        )),
        random_pair,
    ]
}

/// Exact-multiple pairs built around a fixed regression divisor, with a
/// randomized factor and a small offset that is zero on the exact multiple.
fn near_multiple_pair() -> impl Strategy<Value = (InternalMpUint, InternalMpUint)> {
    let random_pair = (
        proptest::collection::vec(any::<Limb>(), 8..=40),
        proptest::collection::vec(any::<Limb>(), 1..=3),
        any::<u8>().prop_map(Limb::from),
    )
        .prop_map(|(den_limbs, factor_limbs, offset)| {
            let den_b = dense_operand(&den_limbs);
            let product = den_b.mul(&dense_operand(&factor_limbs));
            let num_a = product.add(&InternalMpUint::from_limbs(alloc::vec![offset]));
            (num_a, den_b)
        });
    prop_oneof![
        Just((
            dense_operand(&alloc::vec![0xdead; 30])
                .mul(&InternalMpUint::from_limbs(alloc::vec![0x0123])),
            dense_operand(&alloc::vec![0xdead; 30]),
        )),
        random_pair,
    ]
}

/// Builds `num = top * B^(limbs - 1)` and
/// `den = 2 * B^(limbs - 1) - 1`. Thus the leading-limb estimate is `top`,
/// while `top = 8` has exact quotient 4 and `top = 7` has exact quotient 3.
fn maximum_correction_pair(limbs: usize, numerator_top: Limb) -> (InternalMpUint, InternalMpUint) {
    let mut numerator = alloc::vec![0; limbs];
    *numerator
        .last_mut()
        .expect("the correction shape has at least one limb") = numerator_top;
    let mut denominator = alloc::vec![Limb::MAX; limbs];
    *denominator
        .last_mut()
        .expect("the correction shape has at least one limb") = 1;
    (
        InternalMpUint::from_limbs(numerator),
        InternalMpUint::from_limbs(denominator),
    )
}

#[test]
fn small_quotient_classifies_zero_and_one_from_leading_limbs() {
    let limb_len = INLINE_LIMBS;
    let mut unequal_zero_num = alloc::vec![Limb::MAX; limb_len];
    let mut unequal_zero_den = alloc::vec![0; limb_len];
    *unequal_zero_num
        .last_mut()
        .expect("the inline representation is non-empty") = 1;
    *unequal_zero_den
        .last_mut()
        .expect("the inline representation is non-empty") = 2;

    let mut equal_zero_num = alloc::vec![0; limb_len];
    let mut equal_zero_den = alloc::vec![0; limb_len];
    *equal_zero_num
        .last_mut()
        .expect("the inline representation is non-empty") = 3;
    *equal_zero_den
        .last_mut()
        .expect("the inline representation is non-empty") = 3;
    *equal_zero_den
        .first_mut()
        .expect("the inline representation is non-empty") = 1;

    let equal_one_den = equal_zero_den.clone();
    let mut equal_one_num = equal_one_den.clone();
    *equal_one_num
        .first_mut()
        .expect("the inline representation is non-empty") = 2;

    let mut unequal_one_num = alloc::vec![Limb::MAX; limb_len];
    let mut unequal_one_den = alloc::vec![0; limb_len];
    *unequal_one_num
        .last_mut()
        .expect("the inline representation is non-empty") = 3;
    *unequal_one_den
        .last_mut()
        .expect("the inline representation is non-empty") = 2;

    for (numerator_limbs, denominator_limbs) in [
        (unequal_zero_num, unequal_zero_den),
        (equal_zero_num, equal_zero_den),
        (equal_one_num, equal_one_den),
        (unequal_one_num, unequal_one_den),
    ] {
        let numerator = InternalMpUint::from_limbs(numerator_limbs);
        let denominator = InternalMpUint::from_limbs(denominator_limbs);
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        assert!(Division::small_quotient_div_rem(
            &numerator,
            &denominator,
            &mut quotient,
            &mut remainder,
        ));

        let mut algorithm_d_quotient = InternalMpUint::zero();
        let mut algorithm_d_remainder = InternalMpUint::zero();
        let mut scratch = DivScratch::default();
        Division::algorithm_d(
            &numerator,
            &denominator,
            &mut algorithm_d_quotient,
            &mut algorithm_d_remainder,
            &mut scratch,
        );
        assert_eq!(quotient, algorithm_d_quotient);
        assert_eq!(remainder, algorithm_d_remainder);
    }
}

#[test]
fn small_quotient_handles_maximum_corrections_across_inline_boundary() {
    let inline_work_len = INLINE_LIMBS
        .checked_sub(1)
        .expect("the inline representation has a carry slot");
    for limb_len in [inline_work_len, INLINE_LIMBS] {
        for (estimate, exact_quotient) in [(8_u8, 4_usize), (7_u8, 3_usize)] {
            let (numerator, denominator) = maximum_correction_pair(limb_len, Limb::from(estimate));
            let mut quotient = InternalMpUint::zero();
            let mut remainder = InternalMpUint::zero();
            assert!(Division::small_quotient_div_rem(
                &numerator,
                &denominator,
                &mut quotient,
                &mut remainder,
            ));

            let mut algorithm_d_quotient = InternalMpUint::zero();
            let mut algorithm_d_remainder = InternalMpUint::zero();
            let mut scratch = DivScratch::default();
            Division::algorithm_d(
                &numerator,
                &denominator,
                &mut algorithm_d_quotient,
                &mut algorithm_d_remainder,
                &mut scratch,
            );
            assert_eq!(quotient, algorithm_d_quotient);
            assert_eq!(remainder, algorithm_d_remainder);
            assert_eq!(quotient, InternalMpUint::from_limb(exact_quotient));
        }
    }
}

#[test]
fn small_quotient_matches_algorithm_d_around_exact_multiple() {
    let mut denominator_limbs = alloc::vec![3; INLINE_LIMBS];
    *denominator_limbs
        .last_mut()
        .expect("the inline representation is non-empty") = 1;
    let denominator = InternalMpUint::from_limbs(denominator_limbs);
    let factor = InternalMpUint::from_limb(5);
    let exact = denominator.mul(&factor);
    let one = InternalMpUint::one();
    let below = exact.sub(&one);
    let above = exact.add(&one);

    for numerator in [below, exact, above] {
        let mut quotient = InternalMpUint::zero();
        let mut remainder = InternalMpUint::zero();
        assert!(Division::small_quotient_div_rem(
            &numerator,
            &denominator,
            &mut quotient,
            &mut remainder,
        ));

        let mut algorithm_d_quotient = InternalMpUint::zero();
        let mut algorithm_d_remainder = InternalMpUint::zero();
        let mut scratch = DivScratch::default();
        Division::algorithm_d(
            &numerator,
            &denominator,
            &mut algorithm_d_quotient,
            &mut algorithm_d_remainder,
            &mut scratch,
        );
        assert_eq!(quotient, algorithm_d_quotient);
        assert_eq!(remainder, algorithm_d_remainder);
    }
}

#[test]
fn small_quotient_rejection_leaves_outputs_untouched() {
    let mut denominator_limbs = alloc::vec![1; INLINE_LIMBS];
    *denominator_limbs
        .last_mut()
        .expect("the inline representation is non-empty") = 1;
    let denominator = InternalMpUint::from_limbs(denominator_limbs);
    let numerator = denominator.mul(&InternalMpUint::from_limb(9));

    let mut algorithm_d_quotient = InternalMpUint::zero();
    let mut algorithm_d_remainder = InternalMpUint::zero();
    let mut scratch = DivScratch::default();
    Division::algorithm_d(
        &numerator,
        &denominator,
        &mut algorithm_d_quotient,
        &mut algorithm_d_remainder,
        &mut scratch,
    );
    assert_eq!(algorithm_d_quotient, InternalMpUint::from_limb(9));
    assert!(algorithm_d_remainder.is_zero());

    let sentinel_quotient = InternalMpUint::from_limb(0xdead);
    let sentinel_remainder = InternalMpUint::from_limb(0xbeef);
    let mut quotient = sentinel_quotient.clone();
    let mut remainder = sentinel_remainder.clone();
    assert!(!Division::small_quotient_div_rem(
        &numerator,
        &denominator,
        &mut quotient,
        &mut remainder,
    ));
    assert_eq!(quotient, sentinel_quotient);
    assert_eq!(remainder, sentinel_remainder);
}

proptest! {
    /// The truncated quotient matches the full engine whenever it applies,
    /// across quotient lengths that span both the cheap and the correcting
    /// branch, and across the equal-width shapes whose quotient is resolved
    /// by the leading-limb comparison.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest body operates on small usize bounds only"
    )]
    #[test]
    fn prop_truncated_quotient_matches_full_division(
        den_limbs in proptest::collection::vec(any::<Limb>(), 8..=40),
        extra_limbs in proptest::collection::vec(any::<Limb>(), 0..=6),
    ) {
        let den_b = dense_operand(&den_limbs);
        let mut num_limbs = den_limbs;
        num_limbs.extend_from_slice(&extra_limbs);
        let num_a = dense_operand(&num_limbs);

        let mut quot = InternalMpUint::zero();
        if Division::truncated_quotient(&num_a, &den_b, &mut quot) {
            let mut algorithm_d_quotient = InternalMpUint::zero();
            let mut algorithm_d_remainder = InternalMpUint::zero();
            let mut scratch = DivScratch::default();
            Division::algorithm_d(
                &num_a,
                &den_b,
                &mut algorithm_d_quotient,
                &mut algorithm_d_remainder,
                &mut scratch,
            );
            prop_assert_eq!(quot, algorithm_d_quotient);
            prop_assert!(algorithm_d_remainder < den_b);
        }
    }

    /// Dividends adjacent to an exact multiple straddle the sign change of `D`,
    /// which is where an off-by-one correction would surface.  The fixed
    /// regression is an exact multiple of a 30-limb divisor by a single-limb
    /// factor, whose `D` is identically zero.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest body operates on small usize bounds only"
    )]
    #[test]
    fn prop_truncated_quotient_matches_near_exact_multiples(
        pair in near_multiple_pair(),
    ) {
        let (num_a, den_b) = pair;
        let mut quot = InternalMpUint::zero();
        if Division::truncated_quotient(&num_a, &den_b, &mut quot) {
            let mut algorithm_d_quotient = InternalMpUint::zero();
            let mut algorithm_d_remainder = InternalMpUint::zero();
            let mut scratch = DivScratch::default();
            Division::algorithm_d(
                &num_a,
                &den_b,
                &mut algorithm_d_quotient,
                &mut algorithm_d_remainder,
                &mut scratch,
            );
            prop_assert_eq!(quot, algorithm_d_quotient);
            prop_assert!(algorithm_d_remainder < den_b);
        }
    }

    /// Equal-width operands resolve their quotient in the leading-limb
    /// comparison: zero below the divisor, one below the double, and the
    /// truncated machinery above it. Every region must agree with the
    /// full engine.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "proptest body operates on small usize bounds only"
    )]
    #[test]
    fn prop_equal_width_truncated_quotient_matches_full_division(
        pair in equal_width_pair(),
    ) {
        let (num_a, den_b) = pair;
        let mut quot = InternalMpUint::zero();
        if Division::truncated_quotient(&num_a, &den_b, &mut quot) {
            let mut algorithm_d_quotient = InternalMpUint::zero();
            let mut algorithm_d_remainder = InternalMpUint::zero();
            let mut scratch = DivScratch::default();
            Division::algorithm_d(
                &num_a,
                &den_b,
                &mut algorithm_d_quotient,
                &mut algorithm_d_remainder,
                &mut scratch,
            );
            prop_assert_eq!(quot, algorithm_d_quotient);
            prop_assert!(algorithm_d_remainder < den_b);
        }
    }
}
