//! Balanced Toom-Cook 8 and adjacent unbalanced Toom-Cook 8.5 multiplication.
//!
//! - [`cook`]: the two drivers and the tier's split geometry constants.
//! - [`evaluate`] / [`couple`]: the fifteen-point tables and their pairing.
//! - [`interpolate`] / [`linear`]: the inverse system and its linear algebra.
//! - [`layout`]: scratch partition, destination placement, and endpoints.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{
    AddMulKernel, Addition, ArchKernels, KARATSUBA_THRESHOLD, LIMB_BITS, Limb, MulShape,
    Multiplication, Recursive, SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_4_THRESHOLD,
    SQR_TOOM_COOK_6_THRESHOLD, SQR_TOOM_COOK_85_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, SharedEval,
    TOOM_COOK_4_THRESHOLD, TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD,
    TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS, TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS, TierCeiling,
    Widths,
};

// Bound here so every submodule reaches the tier's split geometry and its
// shared product-pair view through `super::`, as the other tiers do.
use cook::{
    BALANCED_PARTS, EVALUATION_GUARD_BITS, HALF_LARGE_PARTS, HALF_SMALL_PARTS,
    INTERPOLATION_GUARD_BITS, ProductPair,
};

mod cook;
mod couple;
mod evaluate;
mod interpolate;
mod layout;
mod linear;

#[cfg(test)]
mod tests;

pub use cook::Toom8;
pub use evaluate::{
    EvaluationDirection, EvaluationPoint, MulEvaluationBuffers, SqrEvaluationBuffers,
};
pub use interpolate::{CouplingContext, Values};
pub use layout::{MulScratchLayout, SqrScratchLayout};
pub use linear::ScaledSource;
