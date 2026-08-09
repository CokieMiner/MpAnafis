//! Toom-Cook 3-by-2 multiplication tier.
//!
//! The first of the fractional-ratio splits. Every balanced Toom tier cuts both
//! operands at one width, so a pair whose lengths differ by a non-integer factor
//! wastes part of every evaluated point; the blocked path repairs integer ratios
//! by turning them into rows of balanced products, but at 1.5 it yields one full
//! block and one half-width block — a balanced product plus a two-to-one one, so
//! the bad shape reappears inside the solution.
//!
//! This tier answers that directly: three parts against two. The product has
//! degree three, so four points suffice, and `{0, 1, -1, infinity}` needs no
//! division beyond a halving. That is a strictly smaller interpolation than
//! Toom-3's five-point solve, which is what the balanced tier would otherwise
//! run on the same operands after zero-extending the shorter one.
//!
//! - [`cook`]: the driver — split, evaluate, recurse, interpolate.
//! - [`evaluate`]: the two-part evaluation and the four-point interpolation.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{
    AddMulKernel, ArchKernels, Limb, Multiplication, Recursive, SharedEval, TierCeiling, Toom3,
    Widths,
};

mod cook;
mod evaluate;

#[cfg(test)]
mod tests;

pub use cook::Toom32;
