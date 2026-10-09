//! Hidden subprocess workers for compiled profiles and adjacent-tier probes.

use super::{CrossoverMeasure, InterleavedMeasure, ProbeQuality};

mod cells;
mod pairs;
mod parsing;
mod profile;
mod selection;

#[cfg(not(target_pointer_width = "16"))]
pub use cells::DIVISION_SCORE_CELLS;
pub use cells::{
    HASH_A, HASH_B, MUL_SCORE_CELLS, PRODUCTION_MUL_CELLS, PRODUCTION_SQR_CELLS, SQR_SCORE_CELLS,
    ScoreCell, TOOM85_MUL_SCORE_CELLS, TOOM85_SQR_SCORE_CELLS, TRANSFORM_SHAPE_CELLS,
};
pub use pairs::PairWorkers;
pub use parsing::{PARSING_CHUNK_SIZES, PARSING_RADICES, ParsingWorker};
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
pub use profile::ParallelWorker;
pub use profile::{
    CONSUMER_FORMAT_RADICES, CONSUMER_SCORE_COUNT, ConsumerWorker, GCD_SCORE_CASES, GcdWorker,
    ProfileWorkers, SsaScoreQuality,
};
#[cfg(not(target_pointer_width = "16"))]
pub use profile::{
    DivisionGrid, DivisionScoreDomain, DivisionWorker, HoldoutWorker, ProductGrid, ProductWorker,
};
pub use selection::CellSelection;

#[cfg(test)]
mod tests;
