//! Cache-oblivious fast Fourier transform and matrix manipulation.
//!
//! Separated into matrix addressing, butterfly passes, and top-level orchestration
//! so target-dependent transpose logic does not crowd the portable transform paths.

mod butterfly;
mod drive;
mod matrix;

pub use drive::SsaTransform;

use super::{
    ArchKernels, FftPlan, Limb, SSA_BASE_MODULUS_BITS, SSA_TRANSPOSE_TILE_LIMBS, SsaCoefficients,
    SsaPointwise, SsaRing,
};
