//! Toom-Cook 4-by-3 multiplication tier.
//!
//! The second fractional-ratio split, covering the band below the three-by-two
//! one: four parts against three admits ratios in `[4/3, 2)`, which is where the
//! measured shape matrix reports its worst cells outside the crossover band.
//!
//! The product has degree five, so six points are needed and
//! `{0, 1, -1, 2, -2, inf}` is the cheapest set that stays on small integers.
//! That is two more recursive products than the three-by-two split spends, and
//! it buys a division by three in the interpolation — the price of reaching a
//! ratio the three-way split cannot express at all.
//!
//! - [`cook`]: the driver — split, evaluate, recurse, interpolate.
//! - [`evaluate`]: paired signed evaluation and the six-point solve.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{
    AddMulKernel, ArchKernels, Limb, Multiplication, Recursive, SharedEval, TierCeiling, Widths,
};

mod cook;
mod evaluate;

#[cfg(test)]
mod tests;

pub use cook::Toom43;
pub use evaluate::{MiddleCoefficients, MiddleProducts};
