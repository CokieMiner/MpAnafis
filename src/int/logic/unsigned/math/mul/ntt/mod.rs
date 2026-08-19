//! Exact multi-prime NTT multiplication with CRT reconstruction.
//!
//! - [`plan`]: the prime table and transform geometry planner.
//! - [`transform`]: the product driver and per-prime workspace orchestration.
//! - [`field`]: field transforms, stage scheduling, and coefficient conversion.
//! - [`butterfly`]: vector-accelerated butterfly stages and pointwise arithmetic.
//! - [`crt`]: two- and three-prime reconstruction into the product.
//! - [`goldilocks`]: the 64-bit Goldilocks field used by the one-prime path.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{ArchKernels, LIMB_BITS, Limb};

use goldilocks::PRIME_U128;
use plan::{MODULI, Modulus};
use transform::{PrimeWorkspace, digit_capacity};

mod butterfly;
mod conversion;
mod crt;
mod field;
mod goldilocks;
mod plan;
mod prepared;
mod transform;

#[cfg(test)]
mod tests;

pub use goldilocks::GoldilocksProduct;
pub use plan::{NttExecutionPolicy, TransformPlan};
pub use prepared::NttMultiplicationPlan;
pub use transform::Ntt;
