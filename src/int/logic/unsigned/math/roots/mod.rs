//! Root operations for [`InternalMpUint`].
//!
//! - [`sqrt`]: square-root basecases and recursive Karatsuba square root.
//! - [`nth`]: general `n`th roots by precision growth.
//! - [`native`]: single-limb roots and double-limb logarithmic estimates.
//! - [`screen`]: residue screening for perfect-square detection.

use super::{
    Addition, ArchKernels, DivScratch, Division, DoubleLimb, INLINE_LIMBS, InternalMpUint,
    LIMB_BITS, Limb, MulScratch, Multiplication, ScratchBuffer,
};

mod native;
mod nth;
mod operations;
mod recursive;
mod screen;
mod seed;
mod sqrt;

pub use operations::NthRootScratch;
pub use screen::Roots;
pub use seed::{EXP2_LOWER, LOG2_LOWER};

#[cfg(test)]
mod tests;
