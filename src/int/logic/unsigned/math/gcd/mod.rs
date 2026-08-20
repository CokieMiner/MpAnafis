//! GCD and LCM operations for [`InternalMpUint`].

use super::{ArchKernels, DivScratch, Division, DoubleLimb, InternalMpUint, LIMB_BITS, Limb};

mod binary;
mod hgcd;
mod lehmer;
mod matrix;
mod operations;

#[cfg(test)]
mod tests;

pub use matrix::HgcdMatrix;
pub use operations::Gcd;
