//! Feature-gated tuning and raw-tier benchmark facade.
//!
//! One internal [`Tuner`] namespace owns the reusable arithmetic runners.
//! Raw slice benchmark operations remain isolated in [`tier`].

#![doc(hidden)]
#![allow(
    unsafe_code,
    reason = "tuner scratch and benchmark shims rely on forced tiers with validated overwrite contracts"
)]

use super::{
    ArchKernels, DivScratch, Division, FormatCache, InternalMpUint, Karatsuba, Lopsided,
    LowProduct, MulScratch, Multiplication, Schoolbook, ScratchBuffer, Ssa, SsaMultiplicationPlan,
    SsaSquaringPlan, Toom3, Toom4, Toom6, Toom8, Toom32, Toom43, TransformBench, TransformChoice,
};

mod division;
mod formatting;
mod multiplication;
mod namespace;
mod squaring;

/// Raw tier runners used only by the dedicated multiplication benchmark.
#[doc(hidden)]
pub mod tier;

pub use division::{DivisionAlgorithm, DivisionRunner};
pub use formatting::{FormattingAlgorithm, FormattingRunner};
pub use multiplication::{MultiplicationAlgorithm, MultiplicationRunner, PreparedMultiplication};
pub use namespace::Tuner;
pub use squaring::{PreparedSquaring, SquaringAlgorithm, SquaringRunner};

pub use super::Limb;
