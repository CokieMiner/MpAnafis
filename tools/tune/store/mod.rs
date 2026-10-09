//! Persistent candidate scores and measurement reports.

use super::{Platform, TuningProfile};

mod context;
mod profile;
mod records;
mod report;

pub use context::MeasurementContext;
pub use profile::ProfileWriter;
pub use records::{SCORE_CACHE_NAME, ScoreStore};

#[cfg(test)]
mod tests;
