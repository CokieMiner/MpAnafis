//! Multiplication tower and tier dispatch.
//!
//! Implemented tiers are partitioned into independent modules with isolated
//! scratch buffer management and aliasing contracts. Schonhage-Strassen (SSA)
//! serves as the terminal quasilinear transform tier.
//!
//! Each conventional tier exposes `mul` and, where defined, `sqr` to execute
//! that tier at the current node. These entry points retain mathematical shape
//! fallbacks. Karatsuba and Toom-3 also expose `dispatch_mul` and `dispatch_sqr`
//! for recursive calls that must consult their configured crossover. Scratch
//! sizing follows the same distinction. Evaluation helpers use `mul_evaluation`
//! and `sqr_evaluation` for the corresponding recursive products.
//!
//! # References
//!
//! - Brent, R. P., & Zimmermann, P. (2011). *Modern Computer Arithmetic*,
//!   Cambridge University Press. <https://doi.org/10.1017/CBO9780511972881>
//! - Bodrato, M., & Zanoni, A. (2007). *Integer and Polynomial Multiplication:
//!   Towards Optimal Toom-Cook Matrices*. ISSAC 2007, 17–24.
//!   <https://doi.org/10.1145/1277548.1277552>.
//! - Schönhage, A., & Strassen, V. (1971). Schnelle Multiplikation großer
//!   Zahlen. *Computing*, 7(3-4), 281–292. <https://doi.org/10.1007/BF02242355>

#[cfg(not(target_pointer_width = "16"))]
use super::{
    AddSubFromKernel, CACHE_BLOCK_BYTES, MUL_MOD_BNM1_THRESHOLD, SQR_SSA_THRESHOLD,
    SSA_BASE_MODULUS_BITS, SSA_BASECASE_COST_WEIGHT_16THS, SSA_BNM1_BASECASE_LIMBS,
    SSA_COEFFICIENT_VISIT_OVERHEAD, SSA_DIRECT_FERMAT_PARALLEL_MIN_WORKERS,
    SSA_DIRECT_FERMAT_PARALLEL_THRESHOLD, SSA_DIRECT_SHIFT_MAX_LIMBS,
    SSA_NEGACYCLIC_FACTOR3_THRESHOLD, SSA_NEGACYCLIC_FACTOR5_THRESHOLD,
    SSA_NESTED_COST_PENALTY_16THS, SSA_PARALLEL_MIN_LIMB_WORK, SSA_SHIFT_BLOCK_WIDTH,
    SSA_SHIFT_SCALAR_THRESHOLD, SSA_THRESHOLD, TRANSFORM_MAX_OPERAND_RATIO,
    TRANSFORM_MIN_SMALLER_LIMBS,
};
use super::{
    Addition, ArchKernels, BALANCED_TOOM8_THRESHOLD, DoubleLimb, INLINE_LIMBS, InternalMpUint,
    KARATSUBA_THRESHOLD, LIMB_BITS, LOPSIDED_TRANSFORM_BLOCK_RATIO, LOW_PRODUCT_FULL_THRESHOLD,
    LOW_PRODUCT_RECURSIVE_THRESHOLD, Limb, SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_4_THRESHOLD,
    SQR_TOOM_COOK_6_THRESHOLD, SQR_TOOM_COOK_85_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, ScratchBuffer,
    TOOM_COOK_4_THRESHOLD, TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD,
    TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS, TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS, UintRepr,
};

mod basecase;
mod dispatch;
mod entry;
mod high;
mod karatsuba;
mod lopsided;
mod low;
mod mulders;
mod recursive;
mod shared;
#[cfg(not(target_pointer_width = "16"))]
mod ssa;
mod toom3;
mod toom32;
mod toom4;
mod toom43;
mod toom6;
mod toom8;

pub use basecase::{LimbOutput, Schoolbook};
#[cfg(not(target_pointer_width = "16"))]
pub use dispatch::LargePlan;
pub use dispatch::{MulPlan, MulShape, Multiplication, SquarePlan, TierCeiling, Widths};
pub use entry::MulScratch;
pub use high::HighProduct;
pub use karatsuba::Karatsuba;
pub use lopsided::Lopsided;
pub use low::LowProduct;
pub use recursive::Recursive;
pub use shared::{AddMulKernel, SharedEval};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use ssa::SsaSquaringPlan;
#[cfg(not(target_pointer_width = "16"))]
pub use ssa::{Ssa, SsaMultiplicationPlan, SsaPlan, TransformChoice};
pub use toom3::Toom3;
pub use toom4::Toom4;
pub use toom6::Toom6;
pub use toom8::Toom8;
pub use toom32::Toom32;
pub use toom43::Toom43;

#[cfg(test)]
mod tests;
