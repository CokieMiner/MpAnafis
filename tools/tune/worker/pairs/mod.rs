//! Adjacent-tier worker protocol and execution.

use super::{CrossoverMeasure, HASH_A, HASH_B, InterleavedMeasure, ProbeQuality, ScoreCell};

mod execution;
mod protocol;

pub use execution::PairWorkers;
pub use protocol::{PairDomain, PairSpecification};

#[cfg(test)]
mod tests;
