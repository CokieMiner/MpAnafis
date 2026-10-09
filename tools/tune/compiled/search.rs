//! Cached candidate screening followed by fresh compiled-profile comparisons.

use crate::worker::{DivisionGrid, DivisionWorker, PARSING_RADICES, TRANSFORM_SHAPE_CELLS};
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use crate::worker::{MUL_SCORE_CELLS, SQR_SCORE_CELLS};

use super::{
    CandidateTrial, CompiledTuner, CoordinateGrid, GCD_SCORE_CASES, Knob, ScoreDomain, TuneSession,
    TuningProfile,
};

/// Cached candidate screening followed by fresh compiled-profile comparisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinateSearch;

impl CoordinateSearch {
    /// Revisit coupled choices in dependency order. The pass limit is a search
    /// budget; reaching it while still changing values does not imply convergence.
    pub fn tune_coordinates(session: &mut TuneSession, name: &str, knobs: &[Knob<'_>]) {
        const MAX_PASSES: usize = 3;
        let before = session.profile;
        let mut grids: Vec<_> = knobs
            .iter()
            .map(|knob| CoordinateGrid::new(knob, &before))
            .collect();
        let mut converged = false;
        for pass in 1..=MAX_PASSES {
            println!("\n{name} coordinate pass {pass}/{MAX_PASSES}");
            let mut changed = false;
            for (knob, grid) in knobs.iter().zip(&mut grids) {
                changed |= Self::tune_knob(session, knob, grid);
                if !session.harness.check_context() {
                    session.profile = before;
                    session.harness.selected_cells.clear();
                    return;
                }
            }
            let expanded: Vec<_> = knobs
                .iter()
                .zip(&grids)
                .map(|(knob, grid)| Knob {
                    candidates: &grid.values,
                    ..*knob
                })
                .collect();
            changed |= Self::tune_pairs(session, &expanded);
            if !changed {
                session.record(name, format!("no accepted change on pass {pass}"));
                converged = true;
                break;
            }
        }
        if !converged {
            session.record(
                name,
                "coordinate budget exhausted; convergence not established",
            );
        }
        let expanded: Vec<_> = knobs
            .iter()
            .zip(&grids)
            .map(|(knob, grid)| Knob {
                candidates: &grid.values,
                ..*knob
            })
            .collect();
        Self::validate_phase(session, name, &before, &expanded);
    }

    /// Rank the full candidate grid, then confirm plausible winners with freshly
    /// executed A/B/B/A and B/A/A/B blocks. Cached scores only determine the
    /// shortlist. Every accepted update beats the current profile directly.
    fn tune_knob(session: &mut TuneSession, knob: &Knob<'_>, grid: &mut CoordinateGrid) -> bool {
        let mut changed = false;
        loop {
            let expanded = Knob {
                candidates: &grid.values,
                ..*knob
            };
            changed |= Self::search_grid(session, &expanded);
            if !session.harness.check_context() || !grid.expand(knob, &session.profile) {
                break;
            }
            session.record(
                format!("{}_GRID_EXPANSION", knob.name),
                format!(
                    "incumbent={}; values={:?}",
                    (knob.get)(session.profile),
                    grid.values
                ),
            );
        }
        changed
    }

    /// Freeze each expanded round's catalog before ranking or confirmation.
    fn search_grid(session: &mut TuneSession, knob: &Knob<'_>) -> bool {
        if !session.harness.check_context() {
            return false;
        }
        let current = (knob.get)(session.profile);
        println!("\nTuning {} (current {current})", knob.name);
        if knob.domain == ScoreDomain::Parsing {
            let profiles = Self::grid_profiles(knob, &session.profile);
            session.harness.parsing_chunks = CompiledTuner::parsing_grid(&profiles, true);
        }
        if knob.domain == ScoreDomain::ProductionDivision {
            let profiles = Self::grid_profiles(knob, &session.profile);
            session.harness.division_grid = CompiledTuner::division_grid(&profiles, true);
        }
        if knob.domain == ScoreDomain::Products {
            let profiles = Self::grid_profiles(knob, &session.profile);
            session.harness.product_grid = CompiledTuner::product_grid(&profiles, true);
        }
        let plan = Self::measurement_plan(session, &[*knob]);
        let mut trials = Vec::new();
        for &value in knob.candidates {
            if value == current {
                continue;
            }
            let mut profile = session.profile;
            (knob.set)(&mut profile, value);
            trials.push(CandidateTrial {
                profile,
                values: vec![value],
            });
        }
        let Some(mut screened) = Self::screen_trials(session, knob.domain, &plan, &trials) else {
            return false;
        };
        let mut anchors = vec![current];
        anchors.extend(
            Self::shortlist_candidates(&mut screened, session.margin_ppm)
                .into_iter()
                .take(3)
                .map(|index| {
                    *trials
                        .get(index)
                        .expect("screened trial")
                        .values
                        .first()
                        .expect("one coordinate")
                }),
        );
        for value in Self::refine_candidates(knob, current, &anchors) {
            if value == current || trials.iter().any(|trial| trial.values == [value]) {
                continue;
            }
            let mut profile = session.profile;
            (knob.set)(&mut profile, value);
            trials.push(CandidateTrial {
                profile,
                values: vec![value],
            });
        }
        let Some(mut refined_scores) = Self::screen_trials(session, knob.domain, &plan, &trials)
        else {
            return false;
        };
        let changed = Self::confirm_trials(session, &[*knob], &plan, &trials, &mut refined_scores);
        println!("Selected {}={}", knob.name, (knob.get)(session.profile));
        changed
    }

    /// Select a policy's code-path objective before observing timings. Complete
    /// phase validation retains guards on outputs omitted from this objective.
    #[must_use]
    pub fn objective_weights(
        knob: &Knob<'_>,
        weights: &[u32],
        division_grid: &DivisionGrid,
        profile: &TuningProfile,
    ) -> Vec<u32> {
        if knob.domain == ScoreDomain::ProductionDivision {
            let maximum = knob
                .candidates
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .max((knob.get)(*profile));
            return DivisionWorker::policy_weights(
                knob.name,
                division_grid,
                maximum,
                profile.newton_quotient.min(profile.newton_raphson),
            );
        }
        let mut objective = weights.to_vec();
        if knob.domain == ScoreDomain::Parsing && knob.name != "RADIX_PARSE_LEAF_CHUNKS" {
            let widths = weights.len().div_euclid(PARSING_RADICES.len());
            for (group, radix) in objective.chunks_exact_mut(widths).zip(PARSING_RADICES) {
                let selected = if knob.name.contains("DECIMAL") {
                    radix == 10
                } else if knob.name.contains("SMALL") {
                    radix < 10
                } else {
                    radix > 10
                };
                if !selected {
                    group.fill(0);
                }
            }
        }
        if knob.name == "BALANCED_TOOM8_THRESHOLD" {
            for (weight, cell) in objective.iter_mut().zip(TRANSFORM_SHAPE_CELLS) {
                if cell.len_a != cell.len_b {
                    *weight = 0;
                }
            }
        }
        if knob.name.starts_with("EXTENDED_") {
            for (index, weight) in objective.iter_mut().enumerate() {
                if !(1..=2).contains(&index.div_euclid(GCD_SCORE_CASES.len())) {
                    *weight = 0;
                }
            }
        }
        #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
        if knob.domain == ScoreDomain::ParallelSsa && knob.name.starts_with("SSA_DIRECT_FERMAT_") {
            let cells = MUL_SCORE_CELLS
                .len()
                .checked_add(SQR_SCORE_CELLS.len())
                .expect("sixteen parallel cells");
            for pool in objective.chunks_exact_mut(cells) {
                pool.iter_mut()
                    .skip(MUL_SCORE_CELLS.len())
                    .for_each(|weight| *weight = 0);
            }
        }
        objective
    }

    /// Rank screened candidates by score and keep the closest cluster.
    #[must_use]
    pub fn shortlist_candidates(screened: &mut [(usize, u128)], margin_ppm: u32) -> Vec<usize> {
        screened.sort_by_key(|&(_, score)| score);
        let Some(&(_, best_score)) = screened.first() else {
            return Vec::new();
        };
        let close_score = best_score.saturating_add(u128::from(margin_ppm).saturating_mul(2));
        screened
            .iter()
            .enumerate()
            .filter_map(|(rank, &(candidate, score))| {
                (rank < 3 || score <= close_score).then_some(candidate)
            })
            .collect()
    }
}
