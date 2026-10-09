//! Cached screening and fresh confirmation of complete candidate profiles.

use crate::validation::VALIDATION_FAMILY_TOLERANCE_PPM;

use super::{
    CandidateHarness, CoordinateSearch, GCD_SCORE_CASES, Knob, MeasurementPlan, SCORE_SCALE,
    ScoreDomain, TuneSession, TuningProfile, Validation,
};

/// One candidate changes all listed coordinates atomically.
#[derive(Clone, Debug)]
pub struct CandidateTrial {
    pub profile: TuningProfile,
    pub values: Vec<usize>,
}

impl CoordinateSearch {
    /// Rank candidates on exactly the same frozen subset as the input profile.
    /// Invalid profiles never reach compilation or operand allocation.
    pub fn screen_trials(
        session: &mut TuneSession,
        domain: ScoreDomain,
        plan: &MeasurementPlan,
        trials: &[CandidateTrial],
    ) -> Option<Vec<(usize, u128)>> {
        let coarse = if domain == ScoreDomain::Ssa {
            ScoreDomain::SsaCoarse
        } else {
            domain
        };
        let baseline = session.harness.score(&session.profile, coarse, false)?;
        let mut screened = Vec::new();
        for (index, trial) in trials.iter().enumerate() {
            if let Err(reason) = trial.profile.validate() {
                println!(
                    "  {}: outside profile constraints: {reason}",
                    trial
                        .values
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                );
                continue;
            }
            let measurements = session.harness.score(&trial.profile, coarse, false)?;
            let score = CandidateHarness::relative_score(&measurements, &baseline, &plan.weights);
            println!(
                "  {}: {score} ppm screening score",
                trial
                    .values
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
            screened.push((index, score));
        }
        Some(screened)
    }

    /// Confirm shortlisted complete profiles against the current incumbent.
    /// Selection is tentative until the complete phase catalog passes its gate.
    pub fn confirm_trials(
        session: &mut TuneSession,
        knobs: &[Knob<'_>],
        plan: &MeasurementPlan,
        trials: &[CandidateTrial],
        screened: &mut [(usize, u128)],
    ) -> bool {
        let domain = knobs.first().expect("candidate coordinates").domain;
        assert_eq!(
            plan.indices.len(),
            plan.weights.len(),
            "one weight per selected index"
        );
        // Rank kernels directly, then confirm the winner together with every
        // consumer guard. Rejected candidates leave the incumbent available
        // for the next shortlist entry rather than losing the entire phase.
        let selection = session.harness.selected_cells.clone();
        let weights = if domain == ScoreDomain::Products {
            session.harness.selected_cells.clear();
            plan.complete_weights(domain.cell_weights(&session.harness).len())
        } else {
            plan.weights.clone()
        };
        let families: Vec<_> = if domain == ScoreDomain::Gcd {
            // The GCD objectives retain complete families, including both
            // extended-GCD and inversion for the shared extended policies.
            (0..plan.weights.len())
                .step_by(GCD_SCORE_CASES.len())
                .map(|start| {
                    start
                        ..start
                            .checked_add(GCD_SCORE_CASES.len())
                            .expect("GCD family span")
                })
                .collect()
        } else {
            Vec::new()
        };
        let rule =
            Validation::comparison_rule(&weights, session.harness.acceptance_limit, &families);
        let mut changed = false;
        for index in Self::shortlist_candidates(screened, session.margin_ppm) {
            let trial = trials.get(index).expect("screened candidate index");
            if trial.profile == session.profile {
                continue;
            }
            let Some(ratios) =
                session
                    .harness
                    .compare_profiles(&session.profile, &trial.profile, domain, &rule)
            else {
                session.harness.selected_cells = selection;
                return false;
            };
            let reference = vec![SCORE_SCALE; weights.len()];
            let score = CandidateHarness::relative_score(&ratios, &reference, &weights);
            let guarded = Validation::cells_non_regress(&ratios, &reference, weights.len());
            let family_guarded = families.iter().all(|span| {
                CandidateHarness::relative_score(
                    ratios.get(span.clone()).expect("worker family"),
                    reference.get(span.clone()).expect("reference family"),
                    plan.weights.get(span.clone()).expect("family weights"),
                ) <= SCORE_SCALE.saturating_add(VALIDATION_FAMILY_TOLERANCE_PPM)
            });
            println!(
                "  {}: {score} ppm upper bound; selected-cell guards {}",
                trial
                    .values
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                guarded && family_guarded
            );
            if guarded && family_guarded && score <= session.harness.acceptance_limit {
                session.profile = trial.profile;
                changed = true;
                for (knob, value) in knobs.iter().zip(&trial.values) {
                    session.record(
                        knob.name,
                        format!(
                            "{value}: tentative upper bound {score} ppm; full phase gate pending"
                        ),
                    );
                }
            }
        }
        session.harness.selected_cells = selection;
        changed
    }
}
