//! Internal implementation layer — unsigned & signed arithmetic, precision.

use super::{
    AmbientPrecision, BoundedPrecision, DoubleLimb, INLINE_LIMBS, LIMB_BITS, LIMB_BYTES, Limb,
};

mod precision;
mod signed;
mod unsigned;

pub use precision::InternalPrecisionContext;
pub use signed::InternalMpInt;
// Production callers reach the arithmetic families through `InternalMpUint`'s
// inherent methods, so this path exists only for the tuning facade and for the
// colocated tests that drive individual tiers directly.
#[cfg(test)]
pub use unsigned::math;
#[cfg(feature = "_internal-tune")]
pub use unsigned::{
    ArchKernels, DivScratch, Division, FormatCache, Karatsuba, Lopsided, LowProduct, MulScratch,
    Multiplication, Schoolbook, ScratchBuffer, Toom3, Toom4, Toom6, Toom8, Toom32, Toom43,
};
pub use unsigned::{InternalMpUint, UintRepr};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use unsigned::{Ssa, SsaMultiplicationPlan, SsaSquaringPlan, TransformBench, TransformChoice};
