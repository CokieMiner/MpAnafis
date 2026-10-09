//! Whole-profile worker protocols and the exact catalogs they execute.

use crate::worker::{
    CONSUMER_SCORE_COUNT, DivisionWorker, GCD_SCORE_CASES, GcdWorker, HoldoutWorker,
    MUL_SCORE_CELLS, PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, ParsingWorker, ProductWorker,
    SQR_SCORE_CELLS, ScoreCell, TOOM85_MUL_SCORE_CELLS, TOOM85_SQR_SCORE_CELLS,
    TRANSFORM_SHAPE_CELLS,
};
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use crate::{platform::Platform, worker::ParallelWorker};

use super::CandidateHarness;

/// Candidate features match the launcher's arithmetic and executor capabilities.
pub const INTERNAL_TUNE_FEATURES: &str = if cfg!(all(feature = "rayon", feature = "num-traits")) {
    "_internal-tune,rayon,num-traits"
} else if cfg!(feature = "rayon") {
    "_internal-tune,rayon"
} else if cfg!(feature = "num-traits") {
    "_internal-tune,num-traits"
} else {
    "_internal-tune"
};

/// Worker domain, independent of the selected indices within its full catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreDomain {
    Ssa,
    SsaCoarse,
    Toom85,
    Toom85Mul,
    ProductionDivision,
    Products,
    Production,
    ProductionShapes,
    Parsing,
    Gcd,
    Consumers,
    Holdout,
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    ParallelSsa,
    #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
    ParallelProduction,
}

impl ScoreDomain {
    /// Freeze geometry arguments and output identity before executing workers.
    pub fn protocol(self, harness: &CandidateHarness) -> (String, &'static str, &'static str) {
        let (base_flag, prefix) = match self {
            Self::Ssa => ("--score-ssa", "MP_ANAFIS_SSA_SCORE="),
            Self::SsaCoarse => ("--score-ssa-coarse", "MP_ANAFIS_SSA_COARSE_SCORE="),
            Self::Toom85 => ("--score-toom85", "MP_ANAFIS_TOOM85_SCORE="),
            Self::Toom85Mul => ("--score-toom85-mul", "MP_ANAFIS_TOOM85_MUL_SCORE="),
            Self::ProductionDivision => (
                "--score-production-division",
                "MP_ANAFIS_PRODUCTION_DIVISION_SCORE=",
            ),
            Self::Production => ("--score-production", "MP_ANAFIS_PRODUCTION_SCORE="),
            Self::Products => ("--score-products", "MP_ANAFIS_PRODUCTS_SCORE="),
            Self::ProductionShapes => (
                "--score-production-shapes",
                "MP_ANAFIS_PRODUCTION_SHAPES_SCORE=",
            ),
            Self::Gcd => ("--score-gcd", "MP_ANAFIS_GCD_SCORE="),
            Self::Parsing => ("--score-parsing", "MP_ANAFIS_PARSING_SCORE="),
            Self::Consumers => ("--score-consumers", "MP_ANAFIS_CONSUMER_SCORE="),
            Self::Holdout => ("--score-holdout", "MP_ANAFIS_HOLDOUT_SCORE="),
            #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
            Self::ParallelSsa => ("--score-parallel-ssa", "MP_ANAFIS_PARALLEL_SSA_SCORE="),
            #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
            Self::ParallelProduction => (
                "--score-parallel-production",
                "MP_ANAFIS_PARALLEL_PRODUCTION_SCORE=",
            ),
        };
        let mut flag = base_flag.to_owned();
        if self == Self::Products {
            flag.push('=');
            flag.push_str(&harness.product_grid.render());
        }
        if self == Self::Parsing {
            flag.push('=');
            flag.push_str(
                &harness
                    .parsing_chunks
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        if matches!(self, Self::ProductionDivision | Self::Production) {
            flag.push('=');
            flag.push_str(&harness.division_grid.render());
        }
        #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
        if matches!(self, Self::ParallelSsa | Self::ParallelProduction) {
            flag.push('=');
            flag.push_str(&harness.parallel_cpus);
            if self == Self::ParallelProduction && !harness.parallel_widths.is_empty() {
                flag.push(';');
                flag.push_str(
                    &harness
                        .parallel_widths
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
        }
        (flag, prefix, INTERNAL_TUNE_FEATURES)
    }

    /// Complete weights match the worker's family and geometry ordering exactly.
    #[must_use]
    pub fn cell_weights(self, harness: &CandidateHarness) -> Vec<u32> {
        match self {
            Self::Ssa | Self::SsaCoarse => {
                ScoreCell::cell_weights(&MUL_SCORE_CELLS, &SQR_SCORE_CELLS)
            }
            Self::Toom85 => {
                ScoreCell::cell_weights(&TOOM85_MUL_SCORE_CELLS, &TOOM85_SQR_SCORE_CELLS)
            }
            Self::Toom85Mul => ScoreCell::cell_weights(&TOOM85_MUL_SCORE_CELLS, &[]),
            Self::ProductionShapes => ScoreCell::cell_weights(&TRANSFORM_SHAPE_CELLS, &[]),
            Self::Gcd => GcdWorker::cell_weights(&GCD_SCORE_CASES),
            Self::Holdout => HoldoutWorker::layout().0,
            Self::Consumers => vec![1; CONSUMER_SCORE_COUNT],
            Self::Products => ProductWorker::cell_weights(&harness.product_grid, None),
            Self::Parsing => ParsingWorker::cell_weights(&harness.parsing_chunks),
            Self::ProductionDivision => {
                DivisionWorker::cell_weights(&DivisionWorker::cases(&harness.division_grid))
            }
            Self::Production => {
                let mut weights =
                    ScoreCell::cell_weights(&PRODUCTION_MUL_CELLS, &PRODUCTION_SQR_CELLS);
                weights.extend(DivisionWorker::cell_weights(&DivisionWorker::cases(
                    &harness.division_grid,
                )));
                weights.extend(GcdWorker::cell_weights(&GCD_SCORE_CASES));
                weights
            }
            #[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
            Self::ParallelSsa | Self::ParallelProduction => {
                let maximum = Platform::parse_cpu_list(&harness.parallel_cpus)
                    .expect("validated parallel CPUs")
                    .len();
                ParallelWorker::cell_weights(
                    maximum,
                    self == Self::ParallelProduction,
                    &harness.parallel_widths,
                )
            }
        }
    }
}
