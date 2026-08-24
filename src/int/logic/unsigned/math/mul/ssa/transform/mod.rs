//! Cache-oblivious fast Fourier transform and matrix manipulation.
//!
//! Separated into matrix addressing, butterfly passes, and top-level orchestration
//! so target-dependent transpose logic does not crowd the portable transform paths.

mod drive;
mod entry;
mod forward;
mod inverse;
mod matrix;

use super::{
    ArchKernels, FftPlan, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, SSA_PARALLEL_MIN_LIMB_WORK,
    SsaCoefficients, SsaPointwise, SsaRing,
};

pub use drive::SsaTransform;
