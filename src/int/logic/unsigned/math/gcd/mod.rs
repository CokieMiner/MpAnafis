//! GCD and LCM operations for [`InternalMpUint`].
//!
//! Modules are declared in call-graph order, grouped by tier: the entry
//! dispatcher, the recursive reduction and the scratch it owns, the quadratic
//! Lehmer reduction, and the in-register leaf kernels last.

use super::{
    Addition, ArchKernels, BINARY_EUCLID_DIVISION_SHIFT, DivScratch, Division, DoubleLimb,
    HGCD_BLOCK_THRESHOLD, HGCD_CROSSOVER_THRESHOLD, InternalMpUint, LEHMER_BRANCHLESS_THRESHOLD,
    LEHMER_FUSED_UPDATE_MAX_LIMBS, LIMB_BITS, Limb, Multiplication, WIDE_LEHMER_THRESHOLD,
};

mod dispatch;
mod jacobi;
mod operations;

mod hgcd;
mod matrix;
mod matrix_update;
mod workspace;

mod hgcd2;
mod lehmer;
mod lehmer_simulation;
mod reduction;

mod binary;

pub use dispatch::Gcd;
pub use lehmer::SignedLimbCarry;
pub use matrix::HgcdMatrix;
pub use workspace::{HgcdFrame, HgcdLocals, HgcdWorkspace};

#[cfg(test)]
mod tests;
