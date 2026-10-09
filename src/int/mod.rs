//! Multi-precision integer subsystem facade and structural module registry.

mod api;
mod logic;
#[cfg(feature = "_internal-tune")]
pub mod tune_api;
mod types;

pub use api::{
    AmbientPrecision, BoundedPrecision, DebugVerbose, MpInt, MpUint, Precision, PrecisionContext,
};
#[cfg(feature = "_internal-tune")]
pub use logic::{
    BarrettDomain, Convert, DivScratch, Division, FormatCache, Gcd, HgcdWorkspace, Karatsuba,
    LowProduct, MontgomeryDomain, MontgomeryScratch, MulPlan, MulScratch, Multiplication,
    RadixParameters, Schoolbook, ScratchBuffer, SquarePlan, TierCeiling, Toom3, Toom4, Toom6,
    Toom8,
};
pub use logic::{InternalMpInt, InternalMpUint, InternalPrecisionContext};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use logic::{Ssa, SsaMultiplicationPlan, SsaSquaringPlan, TransformChoice};
pub use types::{DoubleLimb, INLINE_LIMBS, LIMB_BITS, LIMB_BYTES, Limb};

#[cfg(test)]
mod tests;
