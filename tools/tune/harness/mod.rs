//! Rebuild-worker execution and validated candidate scoring.

use super::{ProbeQuality, ScoreStore, TuningProfile};

mod candidate;
mod comparison;
mod criteria;
mod domain;
mod executable;

pub use candidate::{CandidateHarness, FormattingPairSpec, SCORE_SCALE, TierPairSpec};
pub use criteria::{ComparisonDecision, ComparisonRule, ScoreLimit};
pub use domain::{INTERNAL_TUNE_FEATURES, ScoreDomain};
pub use executable::Executables;

#[cfg(test)]
mod tests;
