//! Raw-limb benchmark facade consumed only by `benches/internal_improvement`.

#![doc(hidden)]

use super::{
    ArchKernels, Karatsuba, Lopsided, LowProduct, MulScratch, Multiplication, Schoolbook, Ssa,
    SsaMultiplicationPlan, Toom3, Toom4, Toom6, Toom8, Toom32, Toom43, TransformBench,
    TransformChoice,
};

use validation::BenchValidation;

/// Raw arithmetic and classical multiplication tier entry points.
pub mod algorithms;
/// Reusable dispatcher and lopsided benchmark state.
pub mod state;
/// Raw transform-tier entry points.
pub mod transform;

mod validation;

pub use super::{Limb, Tuner};
