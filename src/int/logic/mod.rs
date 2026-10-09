//! Signed and unsigned arithmetic with ambient precision resolution.

use super::{
    AmbientPrecision, BoundedPrecision, DoubleLimb, INLINE_LIMBS, LIMB_BITS, LIMB_BYTES, Limb,
};

mod precision;
mod signed;
mod unsigned;

pub use precision::InternalPrecisionContext;
pub use signed::InternalMpInt;
#[cfg(feature = "_internal-tune")]
pub use unsigned::{
    BarrettDomain, Convert, DivScratch, Division, FormatCache, Gcd, HgcdWorkspace, Karatsuba,
    LowProduct, MontgomeryDomain, MontgomeryScratch, MulPlan, MulScratch, Multiplication,
    RadixParameters, Schoolbook, ScratchBuffer, SquarePlan, TierCeiling, Toom3, Toom4, Toom6,
    Toom8,
};
pub use unsigned::{InternalMpUint, UintRepr};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use unsigned::{Ssa, SsaMultiplicationPlan, SsaSquaringPlan, TransformChoice};
