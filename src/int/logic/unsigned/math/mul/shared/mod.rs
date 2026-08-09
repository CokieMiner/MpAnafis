//! Fixed-width evaluation, interpolation, and exact-division helpers shared
//! by every multiplication tier.
//!
//! Toom-Cook evaluation and interpolation work in guarded fixed-width buffers
//! rather than on normalized values, so each routine here propagates its final
//! carry or borrow through the guard instead of reporting it. Where a final
//! carry is deliberately discarded, the comment states the modular argument
//! that makes the truncation exact.
//!
//! [`addsub`] inspects and prepares the buffers, then holds the guarded
//! additions and subtractions that evaluation and reconstruction run on; and
//! [`exact_div`] divides values that are *known* to be divisible, so each
//! routine there is a single low-to-high recurrence rather than a general
//! division. For an odd divisor the quotient is unique modulo the fixed width,
//! which is what makes those exact on two's-complement negative intermediates
//! as well.
//!
//! Imports the two halves share are declared here, so each reaches them through
//! `super` rather than through the parent module.

#![allow(
    unsafe_code,
    reason = "Proven raw-pointer operations on validated buffers"
)]

use super::{Addition, ArchKernels, LIMB_BITS, Limb};

mod addsub;
mod exact_div;

#[cfg(test)]
mod tests;

pub use addsub::{AddMulKernel, SharedEval};
