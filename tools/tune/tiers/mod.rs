//! Tier-crossover tuning policies for conventional multiplication, squaring,
//! transforms, modular exponentiation, and radix formatting.

use super::{
    CrossoverMeasure, Crossovers, FormattingPairSpec, Parameter, ProbeQuality, Request,
    TuneSession, TuningProfile,
};

mod arithmetic;
mod formatting;
mod transforms;
mod walker;

pub use arithmetic::TierTuner;
pub use formatting::FormattingTuner;
pub use walker::{
    Candidate, ITERATIONS, KARATSUBA_SIZES, LARGE_SIZES, TOOM3_SIZES, TOOM4_SIZES, TOOM6_SIZES,
    TOOM85_SIZES, TowerWalker,
};

#[cfg(test)]
mod tests;
