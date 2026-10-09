//! Guarded fixed-width evaluation, interpolation, and exact division.
//!
//! Toom-Cook evaluation and interpolation work in guarded fixed-width buffers
//! rather than on normalized values, so each routine here propagates its final
//! carry or borrow through the guard instead of reporting it. Where a final
//! carry is deliberately discarded, the comment states the modular argument
//! that makes the truncation exact.
//!
//! Buffer preparation establishes the widths used by arithmetic and exact
//! division. Odd divisors have a unique quotient modulo the retained width,
//! including for signed two's-complement intermediates.

use super::{Addition, ArchKernels, LIMB_BITS, Limb};

mod addsub;
mod buffers;
mod exact_div;
mod types;

pub use types::{AddMulKernel, SharedEval};

#[cfg(test)]
mod tests;
