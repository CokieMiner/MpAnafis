//! Unsigned integer implementation: storage representation, arithmetic, bitwise logic, and memory primitives.

use super::{BoundedPrecision, DoubleLimb, INLINE_LIMBS, LIMB_BITS, LIMB_BYTES, Limb};

mod bitwise;
mod cmp;
mod convert;
mod math;
mod memory;
mod properties;
mod storage;

#[cfg(feature = "_internal-tune")]
pub use convert::{Convert, FormatCache, RadixParameters};
pub use math::{
    Addition, ArchKernels, BarrettDomain, BarrettScratch, Division, MulScratch, Multiplication,
    RADIX_FORMAT_DECIMAL_RECURSIVE_THRESHOLD, RADIX_FORMAT_LARGE_RECURSIVE_THRESHOLD,
    RADIX_FORMAT_SMALL_RECURSIVE_THRESHOLD, RADIX_PARSE_DECIMAL_RECURSIVE_THRESHOLD,
    RADIX_PARSE_LARGE_RECURSIVE_THRESHOLD, RADIX_PARSE_LEAF_CHUNKS,
    RADIX_PARSE_SMALL_RECURSIVE_THRESHOLD,
};
#[cfg(feature = "_internal-tune")]
pub use math::{
    DivScratch, Gcd, HgcdWorkspace, Karatsuba, LowProduct, MontgomeryDomain, MontgomeryScratch,
    MulPlan, Schoolbook, SquarePlan, TierCeiling, Toom3, Toom4, Toom6, Toom8,
};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use math::{Ssa, SsaMultiplicationPlan, SsaSquaringPlan, TransformChoice};
pub use memory::ScratchBuffer;
pub use storage::{InternalMpUint, UintRepr};

#[cfg(test)]
mod tests;
