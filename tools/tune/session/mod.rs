//! Stateful host-tuning context shared by every measurement phase.
//!
//! The session owns the candidate, defaults, measurement harness, acceptance
//! margin, decisions, and machine metadata for one search.

use super::{
    Calibration, CandidateHarness, CrossoverMeasure, SCORE_CACHE_NAME, SCORE_SCALE, TuningProfile,
};

mod driver;
mod phases;
mod state;

pub use state::TuneSession;

#[cfg(test)]
mod tests;
