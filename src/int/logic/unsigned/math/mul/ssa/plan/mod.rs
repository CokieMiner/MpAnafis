//! FFT geometry and scratch planning for recursive Fermat-ring SSA.

use super::{
    InverseTwist, LIMB_BITS, Limb, Multiplication, NegacyclicPlan, PointwiseMulPlan,
    PointwiseSquarePlan, ReconstructionBlocks, SSA_BASE_MODULUS_BITS,
    SSA_BASECASE_COST_WEIGHT_16THS, SSA_BNM1_BASECASE_LIMBS, SSA_COEFFICIENT_VISIT_OVERHEAD,
    SSA_NESTED_COST_PENALTY_16THS, SsaPointwise, SsaRing,
};

mod cache;
mod cost;
mod execution;
mod geometry;
mod planner;

pub use cache::CostMemo;
#[cfg(feature = "std")]
pub use cache::RetainedPlanCache;
pub use cost::{SsaOperation, SsaPlan};
pub use execution::{
    MulTransformInput, MulTransformPlan, SquareTransformInput, SquareTransformPlan,
};
pub use geometry::Geometry;
pub use planner::{FftPlan, MAX_COST_RECURSION_DEPTH, RingPeriods};

#[cfg(test)]
mod tests;
