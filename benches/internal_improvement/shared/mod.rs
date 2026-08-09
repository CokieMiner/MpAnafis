//! Operand generation and size ladders shared by every benchmark group.
//!
//! Re-exported flat: benchmarks routinely need one size ladder and one operand
//! constructor together, and a two-segment path for each buys nothing.

pub mod operands;
pub mod sizes;

pub use operands::{operand, operands, operands_pair};
pub use sizes::*;
