//! Toom-Cook 3-by-2 multiplication tier.
//!
//! Three parts times two produce a degree-three polynomial. Evaluations at
//! `0`, `1`, `-1`, and infinity recover its four coefficients; interpolation
//! requires one exact halving.
//!
//! - [`cook`]: the driver, which splits, evaluates, recurses, and interpolates.
//! - [`evaluate`]: the two-part evaluation and the four-point interpolation.

use super::{
    AddMulKernel, ArchKernels, Limb, LimbOutput, Multiplication, Recursive, SharedEval,
    TierCeiling, Toom3, Widths,
};

mod cook;
mod evaluate;

pub use cook::Toom32;

#[cfg(test)]
mod tests;
