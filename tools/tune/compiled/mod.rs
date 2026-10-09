//! Compiled policy searches over frozen operation catalogs.

#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
use super::ParallelWorker;
use super::{
    CandidateHarness, DivisionGrid, DivisionWorker, GCD_SCORE_CASES, Parameter, Platform,
    ProductGrid, ProductWorker, SCORE_SCALE, ScoreDomain, TuneSession, TuningProfile, Validation,
};

mod domains;
mod grid;
mod knobs;
mod plan;
mod refinement;
mod search;
mod trials;

pub use domains::CompiledTuner;
pub use grid::CoordinateGrid;
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
pub use knobs::PARALLEL_KNOBS;
pub use knobs::{
    DIVISION_KNOBS, GCD_KNOBS, Knob, PARSING_KNOBS, PRODUCT_KNOBS, SSA_KNOBS, TOOM_KNOBS,
    TRANSFORM_SHAPE_KNOBS,
};
pub use plan::MeasurementPlan;
pub use search::CoordinateSearch;
pub use trials::CandidateTrial;

#[cfg(test)]
mod tests;
