//! Tier coverage, ceilings, shape validity, and operand-order symmetry.

use alloc::vec::Vec;

use crate::int::logic::unsigned::math::mul::{
    BALANCED_TOOM8_THRESHOLD, KARATSUBA_THRESHOLD, MulPlan, Multiplication, SquarePlan,
    TierCeiling, Widths,
};
#[cfg(not(target_pointer_width = "16"))]
use crate::int::logic::unsigned::math::mul::{LargePlan, SSA_THRESHOLD};

use super::shapes;

const PRODUCT_CEILINGS: [(TierCeiling, &[MulPlan]); 3] = [
    (
        TierCeiling::Toom3,
        &[MulPlan::Schoolbook, MulPlan::Karatsuba, MulPlan::Toom3],
    ),
    (
        TierCeiling::Toom4,
        &[
            MulPlan::Schoolbook,
            MulPlan::Karatsuba,
            MulPlan::Toom3,
            MulPlan::Toom4,
        ],
    ),
    (
        TierCeiling::Toom6,
        &[
            MulPlan::Schoolbook,
            MulPlan::Karatsuba,
            MulPlan::Toom3,
            MulPlan::Toom32,
            MulPlan::Toom43,
            MulPlan::Toom4,
            MulPlan::Toom6,
            MulPlan::Lopsided,
        ],
    ),
];
const SQUARE_CEILINGS: [(TierCeiling, &[SquarePlan]); 3] = [
    (
        TierCeiling::Toom3,
        &[
            SquarePlan::Schoolbook,
            SquarePlan::Karatsuba,
            SquarePlan::Toom3,
        ],
    ),
    (
        TierCeiling::Toom4,
        &[
            SquarePlan::Schoolbook,
            SquarePlan::Karatsuba,
            SquarePlan::Toom3,
            SquarePlan::Toom4,
        ],
    ),
    (
        TierCeiling::Toom6,
        &[
            SquarePlan::Schoolbook,
            SquarePlan::Karatsuba,
            SquarePlan::Toom3,
            SquarePlan::Toom4,
            SquarePlan::Toom6,
        ],
    ),
];

#[test]
fn product_selection_covers_tiers_and_respects_ceilings_shapes_and_order() {
    let mut seen = Vec::new();
    for (larger, smaller) in shapes::products() {
        for ceiling in [
            TierCeiling::Toom3,
            TierCeiling::Toom4,
            TierCeiling::Toom6,
            TierCeiling::Full,
        ] {
            let plan = Multiplication::select_plan(larger, smaller, ceiling);
            assert_eq!(plan, Multiplication::select_plan(smaller, larger, ceiling));
            if let Some((_, allowed)) = PRODUCT_CEILINGS.iter().find(|(bound, _)| *bound == ceiling)
            {
                assert!(
                    allowed.contains(&plan),
                    "{ceiling:?} selected {plan:?} at {larger}x{smaller}"
                );
            }
            let widths = Widths::new(larger, smaller);
            if plan == MulPlan::Toom32 {
                assert!(widths.toom32_suitable(), "three-by-two split exists");
            }
            if plan == MulPlan::Toom43 {
                assert!(widths.toom43_suitable(), "four-by-three split exists");
            }
            if plan == MulPlan::Toom8 {
                assert!(widths.toom8_shape().is_some(), "eight-way split exists");
            }
            #[cfg(not(target_pointer_width = "16"))]
            if SSA_THRESHOLD == 0 {
                assert_ne!(plan, MulPlan::Large(LargePlan::Ssa));
            }
            if !seen.contains(&plan) {
                seen.push(plan);
            }
        }
    }
    for plan in [
        MulPlan::Schoolbook,
        MulPlan::Karatsuba,
        MulPlan::Toom3,
        MulPlan::Toom32,
        MulPlan::Toom43,
        MulPlan::Toom4,
        MulPlan::Toom6,
        MulPlan::Toom8,
        MulPlan::Lopsided,
    ] {
        assert!(seen.contains(&plan), "the grid selects {plan:?}");
    }
    #[cfg(not(target_pointer_width = "16"))]
    if SSA_THRESHOLD != 0 {
        assert!(
            seen.contains(&MulPlan::Large(LargePlan::Ssa)),
            "the grid selects SSA"
        );
    }
    if BALANCED_TOOM8_THRESHOLD != 0 {
        let threshold = BALANCED_TOOM8_THRESHOLD;
        let below = threshold
            .checked_sub(1)
            .expect("enabled crossover is positive");
        assert_ne!(
            Multiplication::select_plan(below, below, TierCeiling::Full),
            MulPlan::Toom8
        );
        assert_eq!(
            Multiplication::select_plan(threshold, threshold, TierCeiling::Full),
            MulPlan::Toom8
        );
        assert_ne!(
            Multiplication::select_plan(threshold, threshold, TierCeiling::Toom6),
            MulPlan::Toom8
        );
    }
}

#[cfg(not(target_pointer_width = "16"))]
#[test]
fn transform_selection_respects_padding_and_its_enabled_crossover() {
    for (larger, smaller) in shapes::products() {
        let plan = Multiplication::select_plan(larger, smaller, TierCeiling::Full);
        if SSA_THRESHOLD == 0 {
            assert_ne!(plan, MulPlan::Large(LargePlan::Ssa));
        } else if larger >= SSA_THRESHOLD {
            if Widths::new(larger, smaller).transform_padding_is_affordable() {
                assert_eq!(plan, MulPlan::Large(LargePlan::Ssa));
            } else if smaller >= KARATSUBA_THRESHOLD {
                assert_eq!(plan, MulPlan::Lopsided);
            }
        }
    }
}

#[test]
fn square_selection_covers_tiers_and_respects_ceilings() {
    let mut seen = Vec::new();
    for len in shapes::widths() {
        for ceiling in [
            TierCeiling::Toom3,
            TierCeiling::Toom4,
            TierCeiling::Toom6,
            TierCeiling::Full,
        ] {
            let plan = Multiplication::select_square_plan(len, ceiling);
            if let Some((_, allowed)) = SQUARE_CEILINGS.iter().find(|(bound, _)| *bound == ceiling)
            {
                assert!(
                    allowed.contains(&plan),
                    "{ceiling:?} selected {plan:?} at {len}^2"
                );
            }
            if !seen.contains(&plan) {
                seen.push(plan);
            }
        }
    }
    for plan in [
        SquarePlan::Schoolbook,
        SquarePlan::Karatsuba,
        SquarePlan::Toom3,
        SquarePlan::Toom4,
        SquarePlan::Toom6,
        SquarePlan::Toom8,
    ] {
        assert!(seen.contains(&plan), "the grid selects {plan:?}");
    }
}
