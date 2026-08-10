//! Power-of-two paired evaluation for the balanced Toom-Cook 6 tier.
//!
//! - [`polynomial`]: the split operand view and its evaluation at one point, as
//!   an even and an odd accumulator.
//! - [`pairs`]: the conjugate-pair products those accumulators feed, and the
//!   five-point schedules the tier runs.

mod pairs;
mod polynomial;

pub use pairs::{MulEvaluationBuffers, SqrEvaluationBuffers};
pub use polynomial::{EvaluationDirection, Parts};

use super::{
    AddMulKernel, ArchKernels, Limb, ProductPair, Recursive, SharedEval, TierCeiling, Toom6, Values,
};
