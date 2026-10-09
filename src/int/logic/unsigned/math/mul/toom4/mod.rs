//! Balanced Toom-Cook 4-way multiplication and squaring.

use super::{
    AddMulKernel, Addition, ArchKernels, DoubleLimb, LIMB_BITS, Limb, LimbOutput, Multiplication,
    Recursive, SharedEval, TierCeiling,
};

mod cook;
mod evaluate;
mod interpolate;
mod paired;

pub use cook::Toom4;
pub use interpolate::MiddleValues;
pub use paired::{
    EvaluationBuffers, EvaluationKernels, MiddleProducts, OperandParts, PointDimensions,
};

#[cfg(test)]
mod tests;
