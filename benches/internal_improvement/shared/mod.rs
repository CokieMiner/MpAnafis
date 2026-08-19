//! Operand generation and size ladders shared by every benchmark group.
//!
//! Re-exported flat: benchmarks routinely need one size ladder and one operand
//! constructor together, and a two-segment path for each buys nothing.

pub mod operands;
pub mod sizes;

#[cfg(feature = "_internal-tune")]
pub use operands::{gmp_equal_reference, gmp_pair_reference};
pub use operands::{operand, operands, operands_pair, validate_and_warm_product};
pub use sizes::*;
