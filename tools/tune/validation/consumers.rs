//! Complete consumer calls and reserved-catalog installation gates.

use super::{
    CONSUMER_SCORE_COUNT, CandidateHarness, HoldoutWorker, SCORE_SCALE, ScoreDomain, TuneSession,
    VALIDATION_AGGREGATE_TOLERANCE_PPM, VALIDATION_FAMILY_TOLERANCE_PPM, Validation,
};

impl Validation {
    /// Confirm complete formatting and modular-power calls against defaults.
    /// Each cell retains its guard in addition to the combined consumer score.
    pub fn consumer_profiles(session: &mut TuneSession) -> bool {
        let weights = vec![1; CONSUMER_SCORE_COUNT];
        let maximum = SCORE_SCALE.saturating_add(VALIDATION_FAMILY_TOLERANCE_PPM);
        session
            .harness
            .compare_profiles(
                &session.defaults,
                &session.profile,
                ScoreDomain::Consumers,
                &Self::comparison_rule(&weights, maximum, &[]),
            )
            .is_some_and(|ratios| {
                let reference = vec![SCORE_SCALE; weights.len()];
                Self::cells_non_regress(&ratios, &reference, reference.len())
                    && CandidateHarness::relative_score(&ratios, &reference, &weights) <= maximum
            })
    }

    /// Freeze the reserved catalog and require aggregate, family and cell
    /// bounds against defaults. No coordinate update uses these observations.
    pub fn holdout_profiles(session: &mut TuneSession) -> bool {
        session.harness.selected_cells.clear();
        let (weights, families) = HoldoutWorker::layout();
        let maximum = SCORE_SCALE.saturating_add(VALIDATION_AGGREGATE_TOLERANCE_PPM);
        let rule = Self::comparison_rule(&weights, maximum, &families);
        println!(
            "\nHeld-out validation: {} cells in {} families",
            weights.len(),
            families.len()
        );
        session.record(
            "HOLDOUT_CATALOG",
            format!(
                "cells={}; families={families:?}; independent widths, seeds and patterns",
                weights.len()
            ),
        );
        let Some(ratios) = session.harness.compare_profiles(
            &session.defaults,
            &session.profile,
            ScoreDomain::Holdout,
            &rule,
        ) else {
            session.record(
                "HOLDOUT_VALIDATION",
                "rejected: reserved worker comparison failed",
            );
            return false;
        };
        let baseline = vec![SCORE_SCALE; weights.len()];
        let valid = Self::cells_non_regress(&ratios, &baseline, weights.len())
            && rule.scores.iter().all(|score| {
                CandidateHarness::relative_score(&ratios, &baseline, &score.weights)
                    <= score.maximum
            });
        session.record(
            "HOLDOUT_VALIDATION",
            format!("accepted={valid}; upper_bounds_ppm={ratios:?}"),
        );
        valid
    }
}
