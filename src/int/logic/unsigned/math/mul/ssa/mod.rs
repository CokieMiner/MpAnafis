//! Recursive Schonhage-Strassen multiplication over Fermat rings.
//!
//! The tier works on flat, caller-owned `&mut [Limb]` coefficient matrices
//! rather than on a `Vec` of bignums. Numeric workspace is partitioned from one
//! caller-owned arena; transform planning can allocate recursive plan metadata.
//! With `std`, bounded per-thread caches retain immutable CRT and nested plans.
//! All Fermat ring arithmetic operates directly on raw limb slices. The parent
//! multiplication registry gates this tier to 32- and 64-bit pointers.
//!
//! # References
//!
//! - Schönhage, A., & Strassen, V. (1971). Schnelle Multiplikation großer
//!   Zahlen. *Computing*, 7(3-4), 281–292. <https://doi.org/10.1007/BF02242355>
//! - Brent, R. P., & Zimmermann, P. (2011). *Modern Computer Arithmetic*,
//!   Section 1.3.5, Cambridge University Press.
//!
//! # Layout
//!
//! - [`entry`]: product and square boundaries, capability predicates, and sizing.
//! - [`mul_plan`], [`square_plan`]: operand-bound plans and scratch execution.
//! - [`plan`]: geometry selection, workspace dimensions, CRT width, and caches.
//! - [`crt`]: the `B^n - 1` half of the top-level split.
//! - [`carry`]: carry and borrow propagation shared across the tier.
//! - [`negacyclic`]: odd-factor decomposition of the Fermat modulus.
//! - [`reconstruct`]: operand to coefficient matrix, and back.
//! - [`transform`]: the butterflies and the matrix addressing.
//! - [`product`]: the pointwise stage.
//! - [`ring`]: arithmetic in `Z/(2^n + 1)`.
//!
//! # Tuning constants
//!
//! Structural limit:
//!
//! - `plan::MAX_COST_RECURSION_DEPTH`: a termination bound on the cost model.
//!
//! Target-dependent tuning constants:
//! - `SSA_BNM1_BASECASE_LIMBS`: where `mul_mod_bnm1` stops splitting. Also a
//!   *correctness* input: [`SsaPlan::crt_half_width`] only emits half-widths whose
//!   odd part fits inside it, which is what keeps the halving recursion off an
//!   odd width. Raising it is always safe; lowering it below the half-widths the
//!   planner emits is not.
//! - `SSA_NEGACYCLIC_FACTOR{3,5}_THRESHOLD`: where an odd factor of the
//!   Fermat modulus repays its extra folds and CRT merge.
//! - `SSA_BASE_MODULUS_BITS`: widest inner ring where pointwise products evaluate
//!   via the base polynomial multiplication tower rather than a nested transform.
//! - `SSA_COEFFICIENT_VISIT_OVERHEAD`: modeled cost of one coefficient visit relative
//!   to single-limb multiplication cost.
//! - `SSA_BASECASE_COST_WEIGHT_16THS`: scales the structural work recurrence
//!   for the lower multiplication tower relative to transform work.
//! - `SSA_DIRECT_SHIFT_MAX_LIMBS`: coefficient width threshold for direct
//!   in-place cyclic shifts.
//! - `SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD`: CRT half-width threshold for concurrent
//!   full-ring transform evaluation.
//! - `SSA_THRESHOLD`: crossover threshold for entering Schonhage-Strassen multiplication.

use super::{
    AddSubFromKernel, Addition, ArchKernels, CACHE_BLOCK_BYTES, DoubleLimb, LIMB_BITS, Limb,
    LimbOutput, MUL_MOD_BNM1_THRESHOLD, MulPlan, MulScratch, Multiplication, SSA_BASE_MODULUS_BITS,
    SSA_BASECASE_COST_WEIGHT_16THS, SSA_BNM1_BASECASE_LIMBS, SSA_COEFFICIENT_VISIT_OVERHEAD,
    SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS, SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD,
    SSA_DIRECT_SHIFT_MAX_LIMBS, SSA_NEGACYCLIC_FACTOR3_THRESHOLD, SSA_NEGACYCLIC_FACTOR5_THRESHOLD,
    SSA_NESTED_COST_PENALTY_16THS, SSA_PARALLEL_MIN_LIMB_WORK, SSA_SHIFT_BLOCK_WIDTH,
    SSA_SHIFT_SCALAR_THRESHOLD, ScratchBuffer, SharedEval, SquarePlan, TierCeiling,
};

mod carry;
mod crt;
mod entry;
mod mul_plan;
mod negacyclic;
mod plan;
mod product;
mod reconstruct;
mod ring;
mod square_plan;
mod transform;
mod two_by_one;

pub use carry::SsaCarry;
pub use crt::{CrtMulPlan, CrtSquarePlan, SsaCrt};
pub use entry::{Ssa, TransformChoice};
pub use mul_plan::SsaMultiplicationPlan;
pub use negacyclic::NegacyclicPlan;
#[cfg(feature = "std")]
pub use plan::RetainedPlanCache;
pub use plan::{
    FftPlan, MulTransformInput, MulTransformPlan, RingPeriods, SquareTransformInput,
    SquareTransformPlan, SsaOperation, SsaPlan,
};
pub use product::{PointwiseMulPlan, PointwiseSquarePlan, SsaPointwise};
pub use reconstruct::{InverseTwist, ReconstructionBlocks, SsaCoefficients};
pub use ring::{Residue, SsaRing};
pub use square_plan::SsaSquaringPlan;
pub use transform::SsaTransform;

#[cfg(test)]
mod tests;
