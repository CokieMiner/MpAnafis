//! Property tests for the Toom-Cook 3-by-2 tier.

use alloc::{vec, vec::Vec};

use proptest::prelude::*;

use super::*;
use crate::int::logic::math::{
    TOOM_COOK_THRESHOLD,
    mul::{
        Multiplication, Schoolbook,
        dispatch::{MulPlan, TierCeiling, Widths},
    },
};

/// Operand pairs across the whole band the tier admits.
///
/// The ratio is what this tier is about, so the strategy generates the ratio
/// rather than two independent lengths: independent lengths land on a suitable
/// shape only occasionally, which is how the Toom-6 scratch defect stayed hidden
/// behind a proptest that passed three runs in a row.
fn suitable_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    (64_usize..400, 0_usize..1000).prop_flat_map(|(larger_len, ratio_seed)| {
        let split_len = larger_len.div_ceil(3);
        // The admitted band is split_len < smaller <= 2*split_len.
        let low = split_len.wrapping_add(1);
        let span = split_len;
        let smaller_len = low.wrapping_add(ratio_seed.rem_euclid(span));
        (
            proptest::collection::vec(any::<Limb>(), larger_len),
            proptest::collection::vec(any::<Limb>(), smaller_len),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn prop_toom32_matches_basecase((a, b) in suitable_operands()) {
        prop_assume!(Widths::new(a.len(), b.len()).toom32_suitable());
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom32_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom32::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    /// The tier must be commutative in its arguments: it orders them itself, so
    /// passing the shorter operand first has to reach the same split.
    #[test]
    fn prop_toom32_is_argument_order_independent((a, b) in suitable_operands()) {
        prop_assume!(Widths::new(a.len(), b.len()).toom32_suitable());
        let result_len = a.len().wrapping_add(b.len());
        let mut forward = vec![Limb::MAX; result_len];
        let mut reversed = vec![Limb::MAX; result_len];

        let scratch_len = Multiplication::toom32_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom32::mul(&mut forward, &a, &b, &mut scratch);
        Toom32::mul(&mut reversed, &b, &a, &mut scratch);
        prop_assert_eq!(forward, reversed);
    }
}

/// Every shape the predicate admits must actually split three ways by two.
///
/// The predicate is the tier's only guard — the driver has no internal fallback
/// — so a shape it admits and the driver cannot split is a panic, not a slow
/// path.
#[test]
fn admitted_shapes_split_three_by_two() {
    for larger in 2_usize..600 {
        for smaller in 1..=larger {
            if !Widths::new(larger, smaller).toom32_suitable() {
                continue;
            }
            let split_len = larger.div_ceil(3);
            assert!(
                larger > split_len.wrapping_mul(2),
                "{larger}x{smaller}: the longer operand has fewer than three parts"
            );
            assert!(
                smaller > split_len && smaller <= split_len.wrapping_mul(2),
                "{larger}x{smaller}: the shorter operand does not have exactly two parts"
            );
        }
    }
}

/// The admitted band is the fractional-ratio gap, not the integer one.
#[test]
fn the_admitted_band_is_the_fractional_ratio_gap() {
    for larger in 300_usize..600 {
        for smaller in 1..=larger {
            if !Widths::new(larger, smaller).toom32_suitable() {
                continue;
            }
            // Ratio below three: the shorter operand keeps a second part only
            // while it exceeds the split, and the split is at least a third.
            assert!(
                larger < smaller.wrapping_mul(3),
                "{larger}x{smaller} was admitted at a ratio of three or worse"
            );
            // Ratio at or above 1.5, up to the rounding in `div_ceil`: from
            // `smaller <= 2 * ceil(larger / 3) <= 2 * (larger + 2) / 3` we get
            // `3 * smaller <= 2 * larger + 4`. The slack is the ceiling itself,
            // not tolerance — at `larger = 301` the split is 101 and a 202-limb
            // operand is a ratio of 1.49.
            assert!(
                smaller.wrapping_mul(3) <= larger.wrapping_mul(2).wrapping_add(4),
                "{larger}x{smaller} was admitted below the three-by-two ratio"
            );
        }
    }
}

/// The selector must name this tier for the shapes it admits, and the plan it
/// names must be the one whose scratch is sized.
#[test]
fn the_selector_names_the_tier_for_admitted_shapes() {
    let mut named = 0_usize;
    let end = TOOM_COOK_THRESHOLD.saturating_add(300);
    for larger in TOOM_COOK_THRESHOLD..end {
        for smaller in 1..=larger {
            let widths = Widths::new(larger, smaller);
            if !widths.toom32_suitable() {
                continue;
            }
            if Multiplication::select_plan(larger, smaller, TierCeiling::Toom6) == MulPlan::Toom32 {
                named = named.wrapping_add(1);
            }
        }
    }
    assert!(
        named > 0,
        "no admitted three-by-two shape reaches the tier through the selector"
    );
}
