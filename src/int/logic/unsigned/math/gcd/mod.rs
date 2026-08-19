//! GCD and LCM operations for [`InternalMpUint`].

use super::{ArchKernels, DivScratch, Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb};

mod binary;
mod lehmer;
mod operations;

#[cfg(test)]
mod tests;

pub use operations::Gcd;
