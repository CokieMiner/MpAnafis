//! Production profile gates and shared regression criteria.
//!
//! Search and final installation use the same aggregate, family, and cell
//! limits. Reserved fixtures participate only in the installation gate.

use super::{
    CONSUMER_SCORE_COUNT, CandidateHarness, ComparisonRule, CompiledTuner, DivisionWorker,
    FormattingPairSpec, GCD_SCORE_CASES, GcdWorker, HoldoutWorker, PRODUCTION_MUL_CELLS,
    PRODUCTION_SQR_CELLS, ProbeQuality, SCORE_SCALE, ScoreCell, ScoreDomain, ScoreLimit,
    TuneSession,
};

mod consumers;
mod conversion;
mod criteria;
#[cfg(all(feature = "rayon", not(target_pointer_width = "16")))]
mod parallel;
mod production;

pub use criteria::{
    VALIDATION_AGGREGATE_TOLERANCE_PPM, VALIDATION_FAMILY_TOLERANCE_PPM, Validation,
};

#[cfg(test)]
mod tests;
