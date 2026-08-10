//! Exact multi-prime NTT multiplication with CRT reconstruction.
//!
//! - [`transform`]: the driver, the prime table, and the transform planner.
//! - [`crt`]: two- and three-prime reconstruction into the product.
//! - [`goldilocks`]: the 64-bit Goldilocks field used by the one-prime path.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{LIMB_BITS, Limb};

// Bound here so `crt` reaches the prime table through `super::`.
use goldilocks::PRIME_U128;
use transform::MODULI;
// Bound here so the tier tests reach the forced-geometry surface through
// `super::`, as they did when the driver lived in this file.
#[cfg(test)]
use transform::TransformPlan;

mod crt;
mod goldilocks;
mod transform;

#[cfg(test)]
mod tests;

pub use transform::Ntt;
