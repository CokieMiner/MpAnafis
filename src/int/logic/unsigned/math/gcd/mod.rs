//! GCD and LCM operations for [`InternalArbiUint`].

use super::{ArchKernels, DivScratch, Division, DoubleLimb, InternalArbiUint, LIMB_BITS, Limb};

mod binary;
mod lehmer;
mod operations;

#[cfg(test)]
mod tests;

pub use operations::Gcd;
