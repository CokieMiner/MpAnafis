//! Feature-gated tuning and raw-tier benchmark facade.
//!
//! Each arithmetic family owns a reusable [`Tuner`](multiplication::Tuner).
//! Raw slice functions for the Divan benchmark remain isolated in [`tier`].

#![doc(hidden)]
#![allow(
    unsafe_code,
    reason = "tuner scratch and benchmark shims rely on forced tiers with validated overwrite contracts"
)]

use super::{
    ArchKernels, DivScratch, Division, FormatCache, InternalArbiUint, Karatsuba, Lopsided,
    LowProduct, MulScratch, Multiplication, Ntt, Schoolbook, ScratchBuffer, Ssa, Toom3, Toom4,
    Toom6, Toom8, Toom32, Toom43, TransformBench, TransformChoice,
};

/// Reusable division crossover tuner.
pub mod division;
/// Reusable formatting crossover tuner.
pub mod formatting;
/// Reusable multiplication crossover tuner.
pub mod multiplication;
/// Reusable squaring crossover tuner.
pub mod squaring;

/// Raw tier runners used only by the dedicated multiplication benchmark.
#[doc(hidden)]
pub mod tier;

pub use super::Limb;
