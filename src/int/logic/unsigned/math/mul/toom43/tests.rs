//! Property tests for the Toom-Cook 4-by-3 tier.

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
/// The ratio is generated rather than left to two independent lengths, so every
/// case lands on a suitable shape instead of occasionally doing so.
fn suitable_operands() -> impl Strategy<Value = (Vec<Limb>, Vec<Limb>)> {
    (64_usize..400, 0_usize..1000).prop_flat_map(|(larger_len, ratio_seed)| {
        let split_len = larger_len.div_ceil(4);
        // The admitted band is 2*split_len < smaller <= 3*split_len.
        let low = split_len.wrapping_mul(2).wrapping_add(1);
        let smaller_len = low.wrapping_add(ratio_seed.rem_euclid(split_len));
        (
            proptest::collection::vec(any::<Limb>(), larger_len),
            proptest::collection::vec(any::<Limb>(), smaller_len),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn prop_toom43_matches_basecase((a, b) in suitable_operands()) {
        prop_assume!(Widths::new(a.len(), b.len()).toom43_suitable());
        let result_len = a.len().wrapping_add(b.len());
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom43_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom43::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prop_toom43_is_argument_order_independent((a, b) in suitable_operands()) {
        prop_assume!(Widths::new(a.len(), b.len()).toom43_suitable());
        let result_len = a.len().wrapping_add(b.len());
        let mut forward = vec![Limb::MAX; result_len];
        let mut reversed = vec![Limb::MAX; result_len];

        let scratch_len = Multiplication::toom43_mul_scratch_len(a.len(), b.len());
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom43::mul(&mut forward, &a, &b, &mut scratch);
        Toom43::mul(&mut reversed, &b, &a, &mut scratch);
        prop_assert_eq!(forward, reversed);
    }

    /// The signed points are where a six-point solve goes wrong, and operands of
    /// all-ones maximise every evaluation guard at once.
    #[test]
    fn prop_toom43_handles_extremal_operands(larger_len in 64_usize..300) {
        let split_len = larger_len.div_ceil(4);
        let smaller_len = split_len.wrapping_mul(3);
        prop_assume!(Widths::new(larger_len, smaller_len).toom43_suitable());
        let a = vec![Limb::MAX; larger_len];
        let b = vec![Limb::MAX; smaller_len];

        let result_len = larger_len.wrapping_add(smaller_len);
        let mut expected = vec![0; result_len];
        let mut actual = vec![Limb::MAX; result_len];
        Schoolbook::mul(&mut expected, &a, &b);

        let scratch_len = Multiplication::toom43_mul_scratch_len(larger_len, smaller_len);
        let mut scratch = vec![Limb::MAX; scratch_len];
        Toom43::mul(&mut actual, &a, &b, &mut scratch);
        prop_assert_eq!(actual, expected);
    }
}

/// Every shape the predicate admits must actually split four ways by three.
#[test]
fn admitted_shapes_split_four_by_three() {
    for larger in 2_usize..600 {
        for smaller in 1..=larger {
            if !Widths::new(larger, smaller).toom43_suitable() {
                continue;
            }
            let split_len = larger.div_ceil(4);
            assert!(
                larger > split_len.wrapping_mul(3),
                "{larger}x{smaller}: the longer operand has fewer than four parts"
            );
            assert!(
                smaller > split_len.wrapping_mul(2) && smaller <= split_len.wrapping_mul(3),
                "{larger}x{smaller}: the shorter operand does not have exactly three parts"
            );
        }
    }
}

/// The two fractional tiers must not both claim a shape without an order, and
/// the selector must reach this one for ratios the three-way split cannot.
#[test]
fn the_selector_reaches_the_four_by_three_band() {
    let mut named = 0_usize;
    let mut below_the_three_way_band = 0_usize;
    let end = TOOM_COOK_THRESHOLD.saturating_add(400);
    for larger in TOOM_COOK_THRESHOLD..end {
        for smaller in 1..=larger {
            let widths = Widths::new(larger, smaller);
            if Multiplication::select_plan(larger, smaller, TierCeiling::Toom6) != MulPlan::Toom43 {
                continue;
            }
            named = named.wrapping_add(1);
            assert!(
                widths.toom43_suitable(),
                "{larger}x{smaller} was named Toom43 but the tier cannot split it"
            );
            // Where both admit the shape the three-way split is offered first,
            // so anything reaching here is outside its band.
            assert!(
                !widths.toom32_suitable(),
                "{larger}x{smaller} reached Toom43 while Toom32 also admits it"
            );
            if larger.wrapping_mul(2) < smaller.wrapping_mul(3) {
                below_the_three_way_band = below_the_three_way_band.wrapping_add(1);
            }
        }
    }
    assert!(
        named > 0,
        "no shape reaches the four-by-three tier through the selector"
    );
    assert!(
        below_the_three_way_band > 0,
        "the four-by-three tier never served a ratio below the three-way band"
    );
}
