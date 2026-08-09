//! In-place memory operations — swap, resize, etc.

use super::{INLINE_LIMBS, InternalMpUint, Limb, UintRepr};

mod arena;
mod inplace;

pub use arena::ScratchBuffer;
