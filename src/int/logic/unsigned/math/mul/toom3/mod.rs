//! Toom-Cook 3-way multiplication and squaring tier.
//!
//! - [`cook`]: the driver — split, evaluate, recurse, interpolate.
//! - [`evaluate`]: the five-point tables and operand evaluation helpers.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{
    AddMulKernel, Addition, ArchKernels, Karatsuba, Limb, Multiplication, Recursive,
    SQR_TOOM_COOK_THRESHOLD, SharedEval, TOOM_COOK_THRESHOLD,
};

mod cook;
mod evaluate;

#[cfg(test)]
mod tests;

pub use cook::Toom3;
pub use evaluate::MiddleValues;
