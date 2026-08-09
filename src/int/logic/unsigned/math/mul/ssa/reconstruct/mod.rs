//! Operand-to-coefficient splitting and coefficient-to-product reconstruction.

#![allow(
    unsafe_code,
    reason = "Direct limb-level splitting and accumulation use validated FFT buffer layouts"
)]

use super::{Addition, ArchKernels, LIMB_BITS, Limb, SharedEval, SsaCarry, SsaRing, SsaTransform};

mod accumulate;
mod split;

pub use accumulate::InverseTwist;
pub use split::SsaCoefficients;
