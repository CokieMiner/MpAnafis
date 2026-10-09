//! Balanced Toom-Cook 8 and adjacent unbalanced Toom-Cook 8.5 multiplication.
//!
//! - [`cook`]: the two drivers and the tier's split geometry constants.
//! - [`evaluate`] / [`couple`]: the fifteen-point tables and their pairing.
//! - [`interpolate`] / [`linear`]: the inverse system and its linear algebra.
//! - [`layout`]: scratch partition, destination placement, and endpoints.
//! - [`demand`]: workspace sizing for the fixed-width evaluation children.

use super::{
    AddMulKernel, Addition, ArchKernels, LIMB_BITS, Limb, LimbOutput, MulShape, Multiplication,
    Recursive, SharedEval, TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS,
    TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS, TierCeiling, Widths,
};

mod cook;
mod couple;
mod demand;
mod evaluate;
mod interpolate;
mod layout;
mod linear;

pub use cook::{EVALUATION_GUARD_BITS, INTERPOLATION_GUARD_LIMBS, ProductPair, Toom8};
pub use couple::{MulEvaluationBuffers, SqrEvaluationBuffers};
pub use demand::ChildDemand;
pub use evaluate::{EvaluationDirection, EvaluationPoint, PointShift};
pub use interpolate::{CouplingContext, Values};
pub use layout::BasePoints;

#[cfg(test)]
mod tests;
