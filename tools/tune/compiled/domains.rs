//! Compiled policy grids in arithmetic dependency order.

use crate::worker::PARSING_CHUNK_SIZES;

use super::{
    CoordinateSearch, DIVISION_KNOBS, DivisionGrid, DivisionWorker, Knob, PARSING_KNOBS,
    PRODUCT_KNOBS, Parameter, Platform, ProductGrid, SSA_KNOBS, ScoreDomain, TuneSession,
    TuningProfile,
};
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use super::{PARALLEL_KNOBS, ParallelWorker, Validation};

/// Tuning drivers for compiled arithmetic, product, and transform policies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledTuner;

impl CompiledTuner {
    /// SSA kernels and their working-set budget share one coordinate loop.
    pub fn tune_ssa(session: &mut TuneSession) {
        let candidates = Self::cache_block_candidates();
        let mut knobs = Vec::from(SSA_KNOBS);
        knobs.push(Knob::new(
            Parameter::CACHE_BLOCK_BYTES,
            &candidates,
            ScoreDomain::Ssa,
        ));
        CoordinateSearch::tune_coordinates(session, "SSA", &knobs);
    }

    /// Measure parallel admission and fork budgets on every available pool width.
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    pub fn tune_parallel(session: &mut TuneSession, specification: &str) {
        let before = session.profile;
        let cpus = Platform::parse_cpu_list(specification).expect("validated parallel CPU set");
        session.harness.parallel_cpus = cpus
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let weights = ParallelWorker::cell_weights(cpus.len(), false, &[]);
        session.record(
            "PARALLEL_CELL_GRID",
            format!(
                "cpus={cpus:?}; workers=1..={}; cells={}",
                cpus.len(),
                weights.len()
            ),
        );
        CoordinateSearch::tune_coordinates(session, "Parallel SSA", &PARALLEL_KNOBS);
        if !Validation::parallel_production(session) {
            session.profile = before;
        }
    }

    /// Tunes division dispatch and recursive geometry against complete
    /// quotient, remainder, and divisibility outputs.
    pub fn tune_division(session: &mut TuneSession) {
        let grid = Self::division_grid(&[session.profile], true);
        let weights = DivisionWorker::cell_weights(&DivisionWorker::cases(&grid));
        session.record(
            "DIVISION_CELL_GRID",
            format!(
                "cells={}; geometry={}; fixtures and outputs scored separately",
                weights.len(),
                grid.render()
            ),
        );
        session.harness.division_grid = grid;
        CoordinateSearch::tune_coordinates(session, "Division", &DIVISION_KNOBS);
    }

    /// Freeze direct-product neighbours, including every proposed refinement.
    #[must_use]
    pub fn product_grid(profiles: &[TuningProfile], include_candidates: bool) -> ProductGrid {
        let mut grid = ProductGrid::default();
        for knob in PRODUCT_KNOBS {
            let widths = if knob.name == "MONTGOMERY_CIOS_MAX_LIMBS" {
                &mut grid.montgomery
            } else {
                &mut grid.cyclic
            };
            for value in profiles.iter().map(|&profile| (knob.get)(profile)).chain(
                knob.candidates
                    .iter()
                    .copied()
                    .filter(|_| include_candidates),
            ) {
                if value == 0 {
                    continue;
                }
                widths.extend([
                    value.saturating_sub(1).max(1),
                    value,
                    value.checked_add(1).expect("finite product policies"),
                ]);
            }
            widths.sort_unstable();
            widths.dedup();
        }
        grid
    }

    /// Freeze boundary probes in each policy's operand dimension. Search also
    /// includes declared candidates; validation includes reference and selected
    /// policies. Zero denotes a disabled strategy and has no positive boundary.
    #[must_use]
    pub fn division_grid(profiles: &[TuningProfile], include_candidates: bool) -> DivisionGrid {
        let mut grid = DivisionGrid::default();
        for knob in DIVISION_KNOBS {
            let values = profiles.iter().map(|&profile| (knob.get)(profile)).chain(
                knob.candidates
                    .iter()
                    .copied()
                    .filter(|_| include_candidates),
            );
            for value in values {
                if value == 0 {
                    continue;
                }
                let neighbours = [
                    value.saturating_sub(2).max(1),
                    value.saturating_sub(1).max(1),
                    value,
                    value.checked_add(1).expect("finite division policies"),
                ];
                match knob.name {
                    "DIVISION_SINGLE_NORMALIZED_PREINVERSE"
                    | "DIVISION_SINGLE_UNNORMALIZED_PREINVERSE"
                    | "DIVISION_BASECASE_QUOTIENT_MAX_LIMBS" => {
                        grid.quotient_widths.extend(neighbours);
                    }
                    "DIVISION_SMALL_QUOTIENT_MAX" => grid.scalar_quotients.extend(neighbours),
                    "NEWTON_SMALL_QUOTIENT_BLOCK_RATIO" | "DIVISION_TRUNCATION_RATIO" => {
                        grid.block_ratios.extend(neighbours);
                    }
                    "NEWTON_RAPHSON_BASECASE_LIMBS" => {
                        grid.divisor_widths.extend(neighbours);
                        grid.quotient_widths.extend(neighbours);
                    }
                    _ => grid.divisor_widths.extend(neighbours),
                }
            }
        }
        for values in [
            &mut grid.divisor_widths,
            &mut grid.quotient_widths,
            &mut grid.scalar_quotients,
            &mut grid.block_ratios,
        ] {
            values.sort_unstable();
            values.dedup();
        }
        grid
    }

    /// Freeze chunk neighbours for the input and candidate parsing profiles.
    /// Every profile comparison receives this same encoded worker grid.
    #[must_use]
    pub fn parsing_grid(profiles: &[TuningProfile], include_candidates: bool) -> Vec<usize> {
        let mut widths = PARSING_CHUNK_SIZES.to_vec();
        for knob in PARSING_KNOBS {
            for value in profiles.iter().map(|&profile| (knob.get)(profile)).chain(
                knob.candidates
                    .iter()
                    .copied()
                    .filter(|_| include_candidates),
            ) {
                widths.extend([
                    value.saturating_sub(1).max(1),
                    value,
                    value.checked_add(1).expect("finite parsing cutoff"),
                ]);
            }
        }
        widths.sort_unstable();
        widths.dedup();
        widths
    }

    /// Quarter/half/full/double L2 budgets, bounded by the measured arena range.
    #[must_use]
    pub fn cache_block_candidates() -> Vec<usize> {
        const MIN_BLOCK: u64 = 16 * 1024;
        const MAX_BLOCK: u64 = 4 * 1024 * 1024;
        const ANCHORS: [u64; 2] = [256 * 1024, 1024 * 1024];
        let mut seeds = Vec::with_capacity(6);
        if let Some(l2) = Platform::l2_cache_bytes() {
            for multiple in [l2.div_euclid(4), l2.div_euclid(2), l2, l2.saturating_mul(2)] {
                seeds.push(multiple.clamp(MIN_BLOCK, MAX_BLOCK));
            }
        }
        seeds.extend_from_slice(&ANCHORS);
        seeds.sort_unstable();
        seeds.dedup();
        seeds
            .into_iter()
            .filter_map(|bytes| usize::try_from(bytes).ok())
            .filter(|&bytes| bytes > 0)
            .collect()
    }
}
