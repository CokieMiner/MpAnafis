//! Unsigned integer implementation — `InternalArbiUint`, storage, math, bitwise, etc.

use super::{DoubleLimb, INLINE_LIMBS, LIMB_BITS, LIMB_BYTES, Limb};

use math::{
    BarrettDomain, BarrettScratch, KARATSUBA_THRESHOLD, RADIX_DECIMAL_RECURSIVE_THRESHOLD,
    RADIX_LARGE_RECURSIVE_THRESHOLD, RADIX_SMALL_RECURSIVE_THRESHOLD,
};

/// Bitwise operations.
mod bitwise;
/// Comparisons.
mod cmp;
/// Type conversions.
mod convert;
/// Number theory and math.
pub mod math;
/// Memory primitives.
pub mod memory;
/// Primitive properties.
mod properties;
/// Unsigned storage definition.
mod storage;

#[cfg(feature = "_internal-tune")]
pub use convert::FormatCache;
pub use math::{ArchKernels, MulScratch};
#[cfg(feature = "_internal-tune")]
pub use math::{
    DivScratch, Division, Karatsuba, Lopsided, LowProduct, Multiplication, Ntt, Schoolbook, Ssa,
    Toom3, Toom4, Toom6, Toom8, Toom32, Toom43, TransformBench, TransformChoice,
};
pub use memory::ScratchBuffer;
pub use storage::{InternalArbiUint, UintRepr};
