//! Toom-Cook 4-by-3 multiplication tier.
//!
//! Fractional-ratio polynomial multiplication tier.
//!
//! A four-by-three chunk decomposition admits asymmetric operand ratios in
//! `[4/3, 2)` where balanced splits would introduce significant zero-padding.
//! The degree-5 product polynomial is evaluated at six points `{0, 1, -1, 2, -2, inf}`,
//! requiring six recursive point products and an exact division by three during interpolation.
//!
//! - [`cook`]: the driver: split, evaluate, recurse, interpolate.
//! - [`evaluate`]: paired signed evaluation and the six-point solve.

use super::{
    AddMulKernel, ArchKernels, Limb, LimbOutput, Multiplication, Recursive, SharedEval,
    TierCeiling, Widths,
};

mod cook;
mod evaluate;

pub use cook::Toom43;
pub use evaluate::{MiddleCoefficients, MiddleProducts};

#[cfg(test)]
mod tests;
