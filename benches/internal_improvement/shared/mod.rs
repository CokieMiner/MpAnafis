//! Shared deterministic inputs, result validation, GMP references, and comparison cases.

#[cfg(target_os = "linux")]
mod cases;
mod gmp;
mod operands;
#[cfg(target_os = "linux")]
mod sizes;
mod validation;

#[cfg(target_os = "linux")]
pub use cases::{
    ShapeWorkerCase, WorkerCase, ambient_workers, parallel_shape_cases, parallel_worker_cases,
    shape_cases, worker_cases,
};
pub use gmp::{assert_gmp_limb_width, gmp_equal_reference, validated_gmp_count};
#[cfg(target_os = "linux")]
pub use gmp::{gmp_pair_reference, validated_gmp_counts};
pub use operands::{operand, operands_pair};
#[cfg(target_os = "linux")]
pub use sizes::{HUGE_SHAPES, PRODUCTION_COMPARE_HUGE_SIZES, PRODUCTION_COMPARE_SIZES, SHAPES};
pub use validation::validate_and_warm_product;
