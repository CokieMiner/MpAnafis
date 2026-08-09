//! Balanced Toom-Cook 4-way multiplication and squaring.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{
    AddMulKernel, Addition, ArchKernels, DoubleLimb, LIMB_BITS, Limb, Multiplication, Recursive,
    SharedEval, TierCeiling,
};

mod cook;
mod evaluate;
mod interpolate;
mod paired;

#[cfg(test)]
mod tests;

// Bound here so the sibling modules reach the driver's guarded products and
// the tier's interpolation view through `super::`.
pub use cook::Toom4;
pub use interpolate::MiddleValues;
pub use paired::{
    EvaluationBuffers, EvaluationKernels, MiddleProducts, OperandParts, PointDimensions,
};
