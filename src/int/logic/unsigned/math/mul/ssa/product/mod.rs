//! Pointwise multiplication, squaring, and basecase product reduction for SSA.

mod basecase;
mod mul;
mod square;

use super::{
    ArchKernels, FftPlan, LIMB_BITS, Limb, MulPlan, Multiplication, NegacyclicPlan, Residue,
    SSA_BASE_MODULUS_BITS, SquarePlan, SsaCarry, SsaRing, SsaTransform, TierCeiling,
};

pub use basecase::SsaPointwise;
