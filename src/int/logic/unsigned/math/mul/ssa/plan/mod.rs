//! FFT geometry and scratch planning for recursive Fermat-ring SSA.

use super::{
    InverseTwist, LIMB_BITS, Limb, SSA_BASE_MODULUS_BITS, SSA_BASECASE_COST_WEIGHT_16THS,
    SSA_BNM1_BASECASE_LIMBS, SSA_COEFFICIENT_VISIT_OVERHEAD, SSA_NESTED_COST_PENALTY_16THS,
    SsaPointwise, SsaRing,
};

use planner::{Geometry, MAX_COST_RECURSION_DEPTH};

mod cost;
mod planner;

#[cfg(test)]
mod tests;

pub use cost::SsaPlan;
pub use planner::FftPlan;
