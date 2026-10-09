//! Toom-Cook 3-way multiplication and squaring tier.
//!
//! - [`cook`]: the driver, which splits, evaluates, recurses, and interpolates.
//! - [`evaluate`]: the five-point tables and operand evaluation helpers.

use super::{
    AddMulKernel, Addition, ArchKernels, Karatsuba, Limb, LimbOutput, Multiplication, Recursive,
    SQR_TOOM_COOK_THRESHOLD, SharedEval, TOOM_COOK_THRESHOLD,
};

mod cook;
mod evaluate;

pub use cook::Toom3;
pub use evaluate::MiddleValues;

#[cfg(test)]
mod tests;
