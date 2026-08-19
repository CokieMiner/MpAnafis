//! Karatsuba multiplication and squaring tier.
//!
//! - [`cook`]: the driver, plus the exact fixed-width specializations.
//! - [`balanced`]: the equal-width difference form and its square.
//! - [`helpers`]: fixed-width evaluation and reconstruction primitives.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{
    Addition, ArchKernels, KARATSUBA_THRESHOLD, Limb, Multiplication, SQR_KARATSUBA_THRESHOLD,
    Schoolbook, SharedEval,
};

mod balanced;
mod cook;
mod helpers;

#[cfg(test)]
mod tests;

pub use cook::Karatsuba;
