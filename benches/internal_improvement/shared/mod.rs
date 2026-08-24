//! Operand generation, case labels, and size ladders shared by benchmark groups.
//!
//! Re-exported flat: benchmarks routinely need one size ladder and one operand
//! constructor together, and a two-segment path for each buys nothing.

pub mod cases;
pub mod operands;
pub mod sizes;

pub use cases::{
    ShapeWorkerCase, WorkerCase, ambient_shape_cases, ambient_worker_cases, parallel_shape_cases,
    parallel_worker_cases,
};
#[cfg(feature = "_internal-tune")]
pub use operands::{
    gmp_equal_reference, gmp_pair_reference, to_gmp_limbs, validated_gmp_count,
    validated_gmp_counts,
};
pub use operands::{operand, operands, operands_pair, validate_and_warm_product};
pub use sizes::*;
