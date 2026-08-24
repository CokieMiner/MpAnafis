//! Addition and subtraction for internal big integers.

use super::{ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};

mod assign;
mod fused;
mod limbs;
mod values;

pub use self::limbs::Addition;
