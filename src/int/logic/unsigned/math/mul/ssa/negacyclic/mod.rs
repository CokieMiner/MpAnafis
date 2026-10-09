//! Odd-factor decomposition for medium Fermat-ring point products.

use super::{
    DoubleLimb, LIMB_BITS, Limb, MulPlan, Multiplication, SSA_NEGACYCLIC_FACTOR3_THRESHOLD,
    SSA_NEGACYCLIC_FACTOR5_THRESHOLD, SharedEval, SsaCarry, SsaPointwise, SsaRing, TierCeiling,
};

mod mul;
mod plan;

pub use plan::NegacyclicPlan;

#[cfg(test)]
mod tests;
