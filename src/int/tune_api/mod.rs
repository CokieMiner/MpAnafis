//! Execution runners for algorithm tuning and benchmark comparisons.
//!
//! Runners retain operands or scratch buffers across calls. Prepared calls
//! validate operand and destination shapes before repeated execution.
//! This interface is unstable and available only with `_internal-tune`.

#![doc(hidden)]

use super::{
    BarrettDomain, Convert, DivScratch, Division, FormatCache, Gcd, HgcdWorkspace, InternalMpUint,
    Karatsuba, LowProduct, MontgomeryDomain, MontgomeryScratch, MulPlan, MulScratch,
    Multiplication, RadixParameters, Schoolbook, ScratchBuffer, SquarePlan, TierCeiling, Toom3,
    Toom4, Toom6, Toom8,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{SsaMultiplicationPlan, SsaSquaringPlan};

#[cfg(not(target_pointer_width = "16"))]
mod cyclic_product;
mod division;
mod formatting;
mod gcd;
mod low_product;
mod modular;
mod multiplication;
mod parsing;
mod result;
mod squaring;
mod tier;

#[cfg(not(target_pointer_width = "16"))]
pub use crate::parallel::{DefaultExecutor, ParallelExecutor};

pub use super::Limb;
#[cfg(not(target_pointer_width = "16"))]
pub use super::{Ssa, TransformChoice};
#[cfg(not(target_pointer_width = "16"))]
pub use cyclic_product::{CyclicProductAlgorithm, CyclicProductRunner};
pub use division::{DivisionAlgorithm, DivisionRunner};
pub use formatting::{FormattingAlgorithm, FormattingRunner};
pub use gcd::{
    GcdAlgorithm, GcdRunner, LehmerSimAlgorithm, LehmerSimRunner, LehmerUpdateAlgorithm,
    LehmerUpdateRunner,
};
pub use low_product::{LowProductAlgorithm, LowProductRunner, PreparedLowProduct};
pub use modular::{ModularPowAlgorithm, ModularPowRunner, MontgomeryProductRunner};
pub use multiplication::{MultiplicationAlgorithm, MultiplicationRunner, PreparedMultiplication};
pub use parsing::{ParsingAlgorithm, ParsingRunner};
pub use result::TuningResult;
pub use squaring::{PreparedSquaring, SquaringAlgorithm, SquaringRunner};
pub use tier::{
    MultiplicationBenchState, PreparedMultiplication as PreparedTierMultiplication,
    PreparedSquaring as PreparedTierSquaring, SquaringBenchState,
};

#[cfg(test)]
mod tests;
