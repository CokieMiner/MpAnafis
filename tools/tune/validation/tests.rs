//! Regression rejection for malformed cells and independently guarded families.

use crate::{harness::ComparisonDecision, measure::RatioEstimate, worker::HoldoutWorker};

use super::{CandidateHarness, GCD_SCORE_CASES, GcdWorker, SCORE_SCALE, Validation};

#[test]
fn gcd_guards_reject_hidden_cell_and_family_regressions() {
    let count = GcdWorker::cell_weights(&GCD_SCORE_CASES).len();
    let baseline = vec![10_000; count];
    assert!(Validation::gcd_non_regresses(&baseline, &baseline));
    let mut isolated = vec![9_000; count];
    *isolated.last_mut().expect("GCD cells are nonempty") = 10_500;
    assert!(!Validation::gcd_non_regresses(&isolated, &baseline));
    let mut family = vec![9_000; count];
    for value in family.iter_mut().take(GCD_SCORE_CASES.len()) {
        *value = 10_200;
    }
    assert!(!Validation::gcd_non_regresses(&family, &baseline));
    assert!(!Validation::gcd_non_regresses(&[], &baseline));
    assert!(!Validation::gcd_non_regresses(&vec![0; count], &baseline));
}

#[test]
fn cell_guards_reject_overflow_instead_of_accepting_saturated_products() {
    assert!(Validation::cells_non_regress(
        &[1_030_000],
        &[SCORE_SCALE],
        1
    ));
    assert!(!Validation::cells_non_regress(
        &[1_030_001],
        &[SCORE_SCALE],
        1
    ));
    let enormous = u128::MAX
        .checked_div(SCORE_SCALE)
        .and_then(|limit| limit.checked_add(1))
        .expect("the nonzero score scale has a representable overflow boundary");
    assert!(!Validation::cells_non_regress(
        &[enormous * 2],
        &[enormous],
        1
    ));
    for (candidate, reference, count) in [
        (&[0][..], &[1][..], 1),
        (&[1][..], &[0][..], 1),
        (&[][..], &[][..], 0),
        (&[1][..], &[1][..], 2),
    ] {
        assert!(!Validation::cells_non_regress(candidate, reference, count));
    }
}

#[test]
fn every_reserved_family_can_reject_a_regression_hidden_by_faster_other_families() {
    let (weights, families) = HoldoutWorker::layout();
    let rule = Validation::comparison_rule(&weights, 1_005_000, &families);
    let baseline = vec![SCORE_SCALE; weights.len()];
    for family in families {
        let mut ratios = vec![900_000; weights.len()];
        ratios
            .get_mut(family)
            .expect("reserved family")
            .fill(1_016_000);
        assert!(CandidateHarness::relative_score(&ratios, &baseline, &weights) <= 1_005_000);
        assert!(Validation::cells_non_regress(
            &ratios,
            &baseline,
            weights.len()
        ));
        let bounds: Vec<_> = ratios
            .into_iter()
            .map(|ratio| RatioEstimate {
                median: ratio,
                lower: ratio,
                upper: ratio,
            })
            .collect();
        assert_eq!(rule.decision(&bounds), ComparisonDecision::Reject);
    }
}
