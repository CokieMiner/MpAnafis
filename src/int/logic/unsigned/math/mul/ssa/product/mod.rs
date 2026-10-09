//! Pointwise multiplication, squaring, and basecase product reduction for SSA.

#[cfg(feature = "std")]
use super::RetainedPlanCache;
use super::{
    ArchKernels, FftPlan, LIMB_BITS, Limb, LimbOutput, MulPlan, MulTransformPlan, Multiplication,
    NegacyclicPlan, Residue, SSA_BASE_MODULUS_BITS, SquarePlan, SquareTransformPlan, SsaCarry,
    SsaPlan, SsaRing, SsaTransform, TierCeiling,
};

mod basecase;
mod mul;
mod pair;
mod plan;
mod square;

pub use basecase::SsaPointwise;
pub use plan::{
    PointwiseMulPlan, PointwiseMulStrategy, PointwiseSquarePlan, PointwiseSquareStrategy,
};

#[cfg(test)]
mod tests;
