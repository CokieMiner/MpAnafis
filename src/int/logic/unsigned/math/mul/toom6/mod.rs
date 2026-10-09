//! Six-way Toom-Cook multiplication and squaring, including the 6.5 split.

use super::{
    AddMulKernel, ArchKernels, Limb, LimbOutput, MulShape, Multiplication, Recursive, SharedEval,
    TierCeiling, Widths,
};

mod cook;
mod half;
mod interpolate;
mod pairs;
mod polynomial;

pub use cook::{ProductPair, ScratchLayout, Toom6};
pub use interpolate::Values;
pub use pairs::{MulEvaluationBuffers, SqrEvaluationBuffers};
pub use polynomial::{EvaluationDirection, Parts, PointShift};

#[cfg(test)]
mod tests;
