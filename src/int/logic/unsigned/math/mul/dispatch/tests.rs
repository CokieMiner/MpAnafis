//! Property tests for the centralized multiplication policy.
//!
//! These assert what the dispatcher promises, not which tier it happens to pick.
//! Deliberately *not* a mirror of the selection chain: a mirror catches
//! accidental drift, but it also pins the ordering, so a deliberate policy
//! change reads as a regression and the mirror has to be edited to say the
//! change was intended — which is the same as having no test.
//!
//! The properties below are the ones that must hold for *any* correct policy:
//!
//! - a named tier can compute the product it was named for;
//! - the plan is what executes, and the workspace is the one it needs;
//! - a ceiling is respected;
//! - selection is deterministic in the operand widths alone.

use alloc::{vec, vec::Vec};

use crate::parallel::SequentialExecutor;

use super::super::{
    BALANCED_TOOM8_THRESHOLD, KARATSUBA_THRESHOLD, Limb, SQR_KARATSUBA_THRESHOLD,
    SQR_TOOM_COOK_THRESHOLD, SSA_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD,
    dispatch::{LargePlan, MulPlan, Multiplication, SquarePlan, TierCeiling, Widths},
};

#[test]
fn measured_balanced_toom8_gate_only_applies_at_the_full_root() {
    if BALANCED_TOOM8_THRESHOLD == 0 {
        return;
    }
    let threshold = BALANCED_TOOM8_THRESHOLD;
    assert_ne!(
        Multiplication::select_plan(
            threshold.saturating_sub(1),
            threshold.saturating_sub(1),
            TierCeiling::Full,
        ),
        MulPlan::Toom8,
    );
    assert_eq!(
        Multiplication::select_plan(threshold, threshold, TierCeiling::Full),
        MulPlan::Toom8,
    );
    assert_ne!(
        Multiplication::select_plan(threshold, threshold, TierCeiling::Toom6),
        MulPlan::Toom8,
    );
}

/// Operand widths spanning every crossover, both sides of each, and the ratios
/// that decide a Toom split shape.
fn probe_widths() -> Vec<usize> {
    let mut widths = Vec::new();
    for threshold in [
        KARATSUBA_THRESHOLD,
        TOOM_COOK_THRESHOLD,
        TOOM_COOK_85_THRESHOLD,
        SQR_KARATSUBA_THRESHOLD,
        SQR_TOOM_COOK_THRESHOLD,
        SSA_THRESHOLD,
    ] {
        for delta in 0..3 {
            widths.push(threshold.saturating_sub(delta));
            widths.push(threshold.saturating_add(delta));
        }
    }
    widths.extend([0, 1, 2, 3, 5, 7, 8, 17, 64, 100, 255, 700, 3_000, 6_000]);
    widths.retain(|width| *width > 0);
    widths.sort_unstable();
    widths.dedup();
    widths
}

/// Ratios that straddle every split-shape test, as `(numerator, denominator)`
/// applied to the larger width.
const SHAPE_RATIOS: [(usize, usize); 8] = [
    (1, 1),
    (17, 18),
    (4, 5),
    (3, 4),
    (2, 3),
    (1, 2),
    (1, 4),
    (1, 16),
];

#[cfg(target_pointer_width = "64")]
const LCG_MUL: Limb = 6_364_136_223_846_793_005;
#[cfg(target_pointer_width = "32")]
const LCG_MUL: Limb = 1_664_525;

#[cfg(target_pointer_width = "64")]
const LCG_ADD: Limb = 1_442_695_040_888_963_407;
#[cfg(target_pointer_width = "32")]
const LCG_ADD: Limb = 1_013_904_223;

fn operand(len: usize, seed: Limb) -> Vec<Limb> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
            state | 1
        })
        .collect()
}

/// Every shape the probe grid describes, larger operand first.
fn probe_shapes() -> Vec<(usize, usize)> {
    let mut shapes = Vec::new();
    for larger in probe_widths() {
        for (numerator, denominator) in SHAPE_RATIOS {
            let smaller = larger
                .saturating_mul(numerator)
                .checked_div(denominator)
                .unwrap_or(larger);
            if smaller > 0 && smaller <= larger {
                shapes.push((larger, smaller));
            }
        }
    }
    shapes.sort_unstable();
    shapes.dedup();
    shapes
}

#[test]
fn the_probe_grid_reaches_every_selectable_tier() {
    // A grid that misses a tier makes every property below vacuously true for
    // it. This asserts the coverage the others rely on, so a future edit to
    // `probe_widths` cannot quietly stop exercising a path.
    //
    // Every ceiling counts, not just `Full`. A tier can be alive purely as a
    // recursive child: with the blocked crossover at four-to-three, no
    // top-level shape reaches Toom-3, because the only widths that would are
    // the ones between `TOOM_COOK_THRESHOLD` and `TOOM_COOK_4_THRESHOLD`, and
    // this profile sets the two equal.
    let mut seen = Vec::new();
    for (larger, smaller) in probe_shapes() {
        for ceiling in [
            TierCeiling::Toom3,
            TierCeiling::Toom4,
            TierCeiling::Toom6,
            TierCeiling::Full,
        ] {
            let plan = Multiplication::select_plan(larger, smaller, ceiling);
            if !seen.contains(&plan) {
                seen.push(plan);
            }
        }
    }

    let mut expected_plans = vec![
        MulPlan::Schoolbook,
        MulPlan::Karatsuba,
        MulPlan::Toom3,
        MulPlan::Toom4,
        MulPlan::Toom6,
        MulPlan::Toom8,
        MulPlan::Lopsided,
    ];
    #[cfg(not(target_pointer_width = "16"))]
    if SSA_THRESHOLD != 0 {
        expected_plans.push(MulPlan::Large(LargePlan::Ssa));
    }

    for expected in expected_plans {
        assert!(
            seen.contains(&expected),
            "the grid never selects {expected:?}"
        );
    }
    assert!(
        probe_shapes().len() > 100,
        "the shape grid collapsed to {} entries",
        probe_shapes().len()
    );
}

#[cfg_attr(miri, ignore = "full tier execution is prohibitively slow under Miri")]
#[test]
fn every_named_tier_computes_the_product_it_was_named_for() {
    // The dispatcher names a transform only after its capability predicate
    // admitted the widths. If that predicate and the entry point ever disagree,
    // the executor writes nothing and the product is silently wrong, so this
    // sweeps the two against each other across every shape in the grid.
    let executor = SequentialExecutor;
    for (larger, smaller) in probe_shapes() {
        let left = operand(larger, 0x1234_5678);
        let right = operand(smaller, 0x8765_4321);

        let plan = Multiplication::select_plan(larger, smaller, TierCeiling::Full);
        let mut produced = vec![0; larger + smaller];
        let mut scratch = vec![0; Multiplication::scratch_len(plan, larger, smaller)];
        Multiplication::execute_plan_with_executor(
            plan,
            &mut produced,
            &left,
            &right,
            &mut scratch,
            &executor,
        );

        // The conventional tower, capped below every transform, is the
        // independent reference: it shares no code with the transform tiers.
        let reference_plan = Multiplication::select_plan(larger, smaller, TierCeiling::Toom6);
        let mut expected = vec![0; larger + smaller];
        let mut reference_scratch =
            vec![0; Multiplication::scratch_len(reference_plan, larger, smaller)];
        Multiplication::execute_plan_with_executor(
            reference_plan,
            &mut expected,
            &left,
            &right,
            &mut reference_scratch,
            &executor,
        );

        assert_eq!(produced, expected, "{plan:?} at {larger}x{smaller}");
    }
}

#[cfg_attr(miri, ignore = "full tier execution is prohibitively slow under Miri")]
#[test]
fn every_named_square_tier_computes_the_square_it_was_named_for() {
    let executor = SequentialExecutor;
    for len in probe_widths() {
        let value = operand(len, 0x1234_5678);

        let plan = Multiplication::select_square_plan(len, TierCeiling::Full);
        let mut produced = vec![0; len * 2];
        let mut scratch = vec![0; Multiplication::square_scratch_len(plan, len)];
        Multiplication::execute_square_plan_with_executor(
            plan,
            &mut produced,
            &value,
            &mut scratch,
            &executor,
        );

        let reference_plan = Multiplication::select_square_plan(len, TierCeiling::Toom6);
        let mut expected = vec![0; len * 2];
        let mut reference_scratch =
            vec![0; Multiplication::square_scratch_len(reference_plan, len)];
        Multiplication::execute_square_plan_with_executor(
            reference_plan,
            &mut expected,
            &value,
            &mut reference_scratch,
            &executor,
        );

        assert_eq!(produced, expected, "{plan:?} at {len}^2");
    }
}

#[cfg_attr(miri, ignore = "full tier execution is prohibitively slow under Miri")]
#[test]
fn the_dispatched_product_matches_the_planned_one() {
    // `uint_mul_limbs_into_slice_scratch` re-derives the plan and may take a
    // stack-scratch shortcut. It must still agree with executing the plan the
    // selector names, or the scratch sizing and the execution have diverged.
    let executor = SequentialExecutor;
    for (larger, smaller) in probe_shapes() {
        let left = operand(larger, 0xdead_beef);
        let right = operand(smaller, 0xfeed_face);

        let plan = Multiplication::select_plan(larger, smaller, TierCeiling::Full);
        let mut planned = vec![0; larger + smaller];
        let mut scratch = vec![0; Multiplication::scratch_len(plan, larger, smaller)];
        Multiplication::execute_plan_with_executor(
            plan,
            &mut planned,
            &left,
            &right,
            &mut scratch,
            &executor,
        );

        let mut dispatched = vec![0; larger + smaller];
        let mut dispatch_scratch = vec![0; Multiplication::scratch_len(plan, larger, smaller)];
        Multiplication::mul_limbs_with_slice_scratch(
            &left,
            &right,
            &mut dispatched,
            &mut dispatch_scratch,
        );

        assert_eq!(dispatched, planned, "{plan:?} at {larger}x{smaller}");
    }
}

#[cfg_attr(miri, ignore = "full tier execution is prohibitively slow under Miri")]
#[test]
fn the_dispatched_square_matches_the_planned_one() {
    let executor = SequentialExecutor;
    for len in probe_widths() {
        let value = operand(len, 0xdead_beef);

        let plan = Multiplication::select_square_plan(len, TierCeiling::Full);
        let mut planned = vec![0; len * 2];
        let mut scratch = vec![0; Multiplication::square_scratch_len(plan, len)];
        Multiplication::execute_square_plan_with_executor(
            plan,
            &mut planned,
            &value,
            &mut scratch,
            &executor,
        );

        let mut dispatched = vec![0; len * 2];
        let mut dispatch_scratch = vec![0; Multiplication::square_scratch_len(plan, len)];
        Multiplication::sqr_limbs_with_slice_scratch(
            &value,
            &mut dispatched,
            &mut dispatch_scratch,
        );

        assert_eq!(dispatched, planned, "{plan:?} at {len}^2");
    }
}

#[test]
fn a_ceiling_is_never_exceeded() {
    // Toom evaluators cap their children so a degenerate root geometry falls to
    // a strictly lower tier instead of redispatching to itself. A ceiling that
    // leaked would be an unbounded recursion, not a slow product.
    for (larger, smaller) in probe_shapes() {
        for ceiling in [TierCeiling::Toom3, TierCeiling::Toom4, TierCeiling::Toom6] {
            let plan = Multiplication::select_plan(larger, smaller, ceiling);
            let exceeds_ceiling = match ceiling {
                TierCeiling::Toom3 => matches!(
                    plan,
                    MulPlan::Toom4
                        | MulPlan::Toom6
                        | MulPlan::Toom8
                        | MulPlan::Lopsided
                        | MulPlan::Large(_)
                ),
                TierCeiling::Toom4 => {
                    matches!(plan, MulPlan::Toom6 | MulPlan::Toom8 | MulPlan::Large(_))
                }
                TierCeiling::Toom6 => matches!(plan, MulPlan::Large(_)),
                TierCeiling::Full => false,
            };
            assert!(
                !exceeds_ceiling,
                "{ceiling:?} selected {plan:?} at {larger}x{smaller}"
            );
        }
    }

    for len in probe_widths() {
        for ceiling in [TierCeiling::Toom3, TierCeiling::Toom4, TierCeiling::Toom6] {
            let plan = Multiplication::select_square_plan(len, ceiling);
            let exceeds_ceiling = match ceiling {
                TierCeiling::Toom3 => matches!(
                    plan,
                    SquarePlan::Toom4
                        | SquarePlan::Toom6
                        | SquarePlan::Toom8
                        | SquarePlan::Large(_)
                ),
                TierCeiling::Toom4 => {
                    matches!(
                        plan,
                        SquarePlan::Toom6 | SquarePlan::Toom8 | SquarePlan::Large(_)
                    )
                }
                TierCeiling::Toom6 => matches!(plan, SquarePlan::Large(_)),
                TierCeiling::Full => false,
            };
            assert!(!exceeds_ceiling, "{ceiling:?} selected {plan:?} at {len}^2");
        }
    }
}

#[test]
fn selection_depends_only_on_the_two_widths_and_is_order_free() {
    // `Widths` orders the pair once. Selecting on the swapped pair must give
    // the same plan, or the dispatcher would be sensitive to which operand the
    // caller happened to pass first.
    for (larger, smaller) in probe_shapes() {
        assert_eq!(
            Multiplication::select_plan(larger, smaller, TierCeiling::Full),
            Multiplication::select_plan(smaller, larger, TierCeiling::Full),
            "{larger}x{smaller} depends on operand order"
        );
    }
}

#[test]
fn a_transform_is_reachable_for_every_shape_above_its_crossover() {
    // The defect this replaces: the transform attempt sat below the Toom split
    // gates, so a shape with no valid Toom split fell to a conventional tier
    // however large the product. A transform has no split, so no *split test*
    // may deny it once its crossover is met.
    //
    // One ratio condition does deny it, deliberately and by measurement rather
    // than by any split: past eight to one the shorter operand is mostly padding
    // in the transform's ring and the blocked path wins outright. Both arms are
    // asserted so the two reasons a shape can miss the transform stay distinct —
    // an excluded shape must land on the blocked path, never on a Toom tier.
    for (larger, smaller) in probe_shapes() {
        if SSA_THRESHOLD == 0 || larger < SSA_THRESHOLD {
            continue;
        }
        let plan = Multiplication::select_plan(larger, smaller, TierCeiling::Full);
        if Widths::new(larger, smaller).transform_padding_is_affordable() {
            assert_eq!(
                plan,
                MulPlan::Large(LargePlan::Ssa),
                "{larger}x{smaller} clears the transform crossover but selected {plan:?}"
            );
        } else {
            assert_eq!(
                plan,
                MulPlan::Lopsided,
                "{larger}x{smaller} is too lopsided to pad, so it must block, but selected {plan:?}"
            );
        }
    }
}

#[test]
fn disabled_crossovers_are_never_selected() {
    // A zero threshold disables its tier outright; it must not be reachable
    // through any shape.
    for (larger, smaller) in probe_shapes() {
        let plan = Multiplication::select_plan(larger, smaller, TierCeiling::Full);
        #[cfg(not(target_pointer_width = "16"))]
        if SSA_THRESHOLD == 0 {
            assert_ne!(
                plan,
                MulPlan::Large(LargePlan::Ssa),
                "{larger}x{smaller} selected a disabled tier"
            );
        }
    }
}

#[test]
fn a_toom8_plan_is_only_named_for_a_shape_it_can_split() {
    // Toom-8.5 has no lower fallback inside its own tier, so naming it for a
    // shape with no eight-way split would send the product back through the
    // dispatcher from inside the algorithm.
    for (larger, smaller) in probe_shapes() {
        if Multiplication::select_plan(larger, smaller, TierCeiling::Full) == MulPlan::Toom8 {
            assert!(
                Widths::new(larger, smaller).toom8_shape().is_some(),
                "{larger}x{smaller} was given Toom-8 without an eight-way split"
            );
        }
    }
}
