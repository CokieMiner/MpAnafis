//! In-place memory operations: reallocations, scratch buffer pools, and raw memory guards.

use super::{INLINE_LIMBS, InternalMpUint, Limb, UintRepr};

mod arena;
#[cfg(feature = "std")]
mod bucket;
mod inplace;

pub use arena::ScratchBuffer;
#[cfg(feature = "std")]
pub use bucket::{BucketSlot, MAX_PER_BUCKET};

#[cfg(test)]
mod tests;
