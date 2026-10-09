//! Karatsuba multiplication and squaring tier.
//!
//! - [`cook`]: the driver, plus the exact fixed-width specializations.
//! - [`balanced`]: the equal-width difference form and its square.
//! - [`helpers`]: fixed-width evaluation and reconstruction primitives.

use super::{
    Addition, ArchKernels, KARATSUBA_THRESHOLD, Limb, LimbOutput, Multiplication,
    SQR_KARATSUBA_THRESHOLD, Schoolbook, SharedEval,
};

mod balanced;
mod cook;
mod helpers;

pub use cook::Karatsuba;

#[cfg(test)]
mod tests;
