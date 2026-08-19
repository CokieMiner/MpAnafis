//! Cache-oblivious fast Fourier transform and matrix manipulation.
//!
//! Separated into matrix addressing, butterfly passes, and top-level orchestration
//! so target-dependent transpose logic does not crowd the portable transform paths.

mod drive;
mod entry;
mod forward;
mod inverse;
mod matrix;

use entry::{can_fork_four, should_parallelize};
use forward::{fft_recursive_dif_with_executor, recurse_dif_pair};
use inverse::fft_recursive_dit_with_executor;

use super::{
    ArchKernels, FftPlan, Limb, SSA_BASE_MODULUS_BITS, SsaCoefficients, SsaPointwise, SsaRing,
};

pub use drive::SsaTransform;
