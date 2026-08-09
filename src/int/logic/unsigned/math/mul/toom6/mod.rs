//! Six-way Toom-Cook multiplication and squaring, including the 6.5 split.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{
    AddMulKernel, ArchKernels, Limb, MulShape, Multiplication, Recursive, SharedEval, TierCeiling,
    Widths,
};

// Bound here so the sibling modules reach each other and the tier's shared
// product-pair view through `super::`.
use cook::{ProductPair, ScratchLayout};

mod cook;
mod evaluate;
mod half;
mod interpolate;

#[cfg(test)]
mod tests;

pub use cook::Toom6;
pub use evaluate::{MulEvaluationBuffers, Parts, SqrEvaluationBuffers};
pub use interpolate::Values;
