//! Frozen coordinate objectives and complete phase regression gates.

use crate::{validation::VALIDATION_AGGREGATE_TOLERANCE_PPM, worker::CellSelection};

use super::{
    CandidateHarness, CompiledTuner, CoordinateSearch, GCD_SCORE_CASES, Knob, ProductWorker,
    SCORE_SCALE, ScoreDomain, TuneSession, TuningProfile, Validation,
};

/// The executed indices and their weights share the complete domain ordering.
#[derive(Clone, Debug)]
pub struct MeasurementPlan {
    pub indices: Vec<usize>,
    pub weights: Vec<u32>,
}

impl MeasurementPlan {
    /// Places a sparse kernel objective in the complete confirmation catalog.
    /// Zero-weight consumer cells retain their individual regression guards.
    pub fn complete_weights(&self, count: usize) -> Vec<u32> {
        assert_eq!(
            self.indices.len(),
            self.weights.len(),
            "one weight per objective cell"
        );
        let mut weights = vec![0; count];
        for (&index, &weight) in self.indices.iter().zip(&self.weights) {
            *weights
                .get_mut(index)
                .expect("objective belongs to complete catalog") = weight;
        }
        weights
    }
}

impl CoordinateSearch {
    /// Include every admissible grid and refinement value before constructing
    /// boundary probes. Expanded endpoints require both their neighbours too.
    #[must_use]
    pub fn grid_profiles(knob: &Knob<'_>, profile: &TuningProfile) -> Vec<TuningProfile> {
        let current = (knob.get)(*profile);
        let mut anchors = knob.candidates.to_vec();
        anchors.push(current);
        let mut values = anchors.clone();
        values.extend(Self::refine_candidates(knob, current, &anchors));
        values.sort_unstable();
        values.dedup();
        let mut profiles = vec![*profile];
        for value in values {
            if value == current {
                continue;
            }
            let mut trial = *profile;
            (knob.set)(&mut trial, value);
            if trial.validate().is_ok() {
                profiles.push(trial);
            }
        }
        profiles
    }

    /// Select the union of code-path objectives before observing either binary.
    /// A full selection uses an empty protocol mask; sparse indices retain their
    /// catalog identity in both the cache key and raw worker captures.
    pub fn measurement_plan(session: &mut TuneSession, knobs: &[Knob<'_>]) -> MeasurementPlan {
        let domain = knobs.first().expect("at least one coordinate").domain;
        assert!(
            knobs.iter().all(|knob| knob.domain == domain),
            "one domain per comparison"
        );
        let weights = domain.cell_weights(&session.harness);
        let mut objective = vec![0; weights.len()];
        for knob in knobs {
            let owned = if domain == ScoreDomain::Products {
                ProductWorker::cell_weights(&session.harness.product_grid, Some(knob.name))
            } else {
                Self::objective_weights(
                    knob,
                    &weights,
                    &session.harness.division_grid,
                    &session.profile,
                )
            };
            assert_eq!(
                owned.len(),
                objective.len(),
                "objective matches worker catalog"
            );
            for (target, value) in objective.iter_mut().zip(owned) {
                *target = (*target).max(value);
            }
        }
        let indices: Vec<_> = objective
            .iter()
            .enumerate()
            .filter_map(|(index, &weight)| (weight > 0).then_some(index))
            .collect();
        assert!(!indices.is_empty(), "every coordinate owns measured cells");
        let selected = indices
            .iter()
            .map(|&index| *objective.get(index).expect("selected index"))
            .collect();
        session.harness.selected_cells = if indices.len() == weights.len() {
            Vec::new()
        } else {
            indices.clone()
        };
        session.record(
            "COORDINATE_CELL_PLAN",
            format!(
                "policies={}; selected={}/{}; indices={}",
                knobs
                    .iter()
                    .map(|knob| knob.name)
                    .collect::<Vec<_>>()
                    .join(","),
                indices.len(),
                weights.len(),
                CellSelection::encode(&indices)
            ),
        );
        MeasurementPlan {
            indices,
            weights: selected,
        }
    }

    /// Compare the complete domain after tentative coordinate and pair updates.
    /// A failed gate restores the phase input, including coupled field changes.
    pub fn validate_phase(
        session: &mut TuneSession,
        name: &str,
        before: &TuningProfile,
        knobs: &[Knob<'_>],
    ) {
        session.harness.selected_cells.clear();
        if session.profile == *before {
            return;
        }
        let mut domains = Vec::new();
        for knob in knobs {
            if !domains.contains(&knob.domain) {
                domains.push(knob.domain);
            }
        }
        let mut passed = session.harness.check_context();
        for domain in domains {
            if !passed {
                break;
            }
            if domain == ScoreDomain::Parsing {
                session.harness.parsing_chunks =
                    CompiledTuner::parsing_grid(&[*before, session.profile], true);
            }
            if domain == ScoreDomain::ProductionDivision {
                session.harness.division_grid =
                    CompiledTuner::division_grid(&[*before, session.profile], true);
            }
            if domain == ScoreDomain::Products {
                session.harness.product_grid =
                    CompiledTuner::product_grid(&[*before, session.profile], true);
            }
            let weights = domain.cell_weights(&session.harness);
            let families: Vec<_> = if domain == ScoreDomain::Gcd {
                (0..weights.len())
                    .step_by(GCD_SCORE_CASES.len())
                    .map(|start| {
                        start
                            ..start
                                .checked_add(GCD_SCORE_CASES.len())
                                .expect("GCD family span")
                    })
                    .collect()
            } else if domain == ScoreDomain::Parsing {
                let count = session.harness.parsing_chunks.len();
                (0..weights.len())
                    .step_by(count)
                    .map(|start| start..start.checked_add(count).expect("parsing radix span"))
                    .collect()
            } else {
                Vec::new()
            };
            let rule = Validation::comparison_rule(
                &weights,
                SCORE_SCALE.saturating_add(VALIDATION_AGGREGATE_TOLERANCE_PPM),
                &families,
            );
            passed = session
                .harness
                .compare_profiles(before, &session.profile, domain, &rule)
                .is_some_and(|ratios| {
                    let baseline = vec![SCORE_SCALE; weights.len()];
                    Validation::cells_non_regress(&ratios, &baseline, weights.len())
                        && rule.scores.iter().all(|score| {
                            CandidateHarness::relative_score(&ratios, &baseline, &score.weights)
                                <= score.maximum
                        })
                        && CandidateHarness::relative_score(&ratios, &baseline, &weights)
                            <= SCORE_SCALE.saturating_add(VALIDATION_AGGREGATE_TOLERANCE_PPM)
                        && (domain != ScoreDomain::Gcd
                            || Validation::gcd_non_regresses(&ratios, &baseline))
                });
        }
        if !passed {
            session.profile = *before;
        }
        session.record(
            format!("{name}_PHASE_VALIDATION"),
            if passed {
                "complete catalog passed"
            } else {
                "input profile restored; complete catalog failed or was unresolved"
            },
        );
    }
}
