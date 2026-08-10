//! In-place memory operations — swap, resize, etc.

use super::{INLINE_LIMBS, InternalArbiUint, Limb, UintRepr};

mod arena;
mod inplace;

pub use arena::ScratchBuffer;
