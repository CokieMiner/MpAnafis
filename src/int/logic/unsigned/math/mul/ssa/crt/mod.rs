//! Chinese Remainder Theorem modulo $B^n - 1$ and associated folding utilities.
//!
//! The top-level entry points pair one `B^n + 1` transform with one `B^n - 1`
//! product and merge the two residues. [`bnm1`] carries the product and square
//! recursions with their reconstructions and exact-width staging,
//! [`layout`] sizes every scratch partition for the executor that will run it,
//! and [`two_by_one`] holds the shared-operand recursion that transforms the
//! common operand once per ring.

#[cfg(feature = "std")]
use super::RetainedPlanCache;
use super::{
    Addition, ArchKernels, FftPlan, LIMB_BITS, Limb, LimbOutput, MulPlan, MulTransformPlan,
    Multiplication, SSA_BASE_MODULUS_BITS, SSA_BNM1_BASECASE_LIMBS, SquarePlan,
    SquareTransformPlan, SsaCarry, SsaPointwise, SsaTransform, TierCeiling,
};

mod bnm1;
mod execution;
mod layout;
mod two_by_one;

pub use bnm1::SsaCrt;
pub use execution::{CrtMulLevel, CrtMulPlan, CrtSquareLevel, CrtSquarePlan};

#[cfg(test)]
mod tests;
