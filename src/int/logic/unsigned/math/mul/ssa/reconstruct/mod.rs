//! Operand-to-coefficient splitting and coefficient-to-product reconstruction.

use super::{
    Addition, LIMB_BITS, Limb, LimbOutput, RingPeriods, SSA_PARALLEL_MIN_LIMB_WORK, SharedEval,
    SsaCarry, SsaRing, SsaTransform,
};

mod accumulate;
mod blocks;
mod limbs;
mod split;

pub use accumulate::InverseTwist;
pub use blocks::ReconstructionBlocks;
pub use split::SsaCoefficients;
