//! Paired timing, median-ratio bounds, and crossover searches.
//!
//! Coarse probes screen candidates. Precise probes use a discarded pilot and
//! fresh confirmation samples at predeclared observation counts.

mod confidence;
mod crossover;
mod interleaved;

pub use confidence::{PairedStatistics, RatioEstimate};
pub use crossover::{CrossoverMeasure, MIN_MARGIN_PPM, ProbeQuality};
pub use interleaved::InterleavedMeasure;

#[cfg(test)]
mod tests;
