//! Root operations for [`InternalMpUint`].
//!
//! - [`sqrt`]: square-root basecases and recursive Karatsuba square root.
//! - [`nth`]: general `n`th roots and the single-limb fast path.
//! - [`screen`]: residue screening for perfect-square detection.

use super::{DivScratch, Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb, MulScratch};

use self::screen::may_be_square;

mod nth;
mod operations;
mod screen;
mod sqrt;

#[cfg(test)]
mod tests;

pub use operations::{NthRootScratch, SqrtScratch};
