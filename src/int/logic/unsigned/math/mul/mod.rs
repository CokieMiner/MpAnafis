//! Multiplication tower and tier dispatch.
//!
//! Implemented tiers are kept in independent modules so each algorithm can
//! evolve, benchmark, and prove its own scratch invariants without growing a
//! monolithic multiplication file. SSA is the active transform tier; NTT/CRT
//! remains registered for development and benchmark work but is disabled by its
//! generated threshold.

use super::{
    Addition, ArchKernels, DoubleLimb, INLINE_LIMBS, InternalMpUint, KARATSUBA_THRESHOLD,
    LIMB_BITS, Limb, NTT_THRESHOLD, SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_4_THRESHOLD,
    SQR_TOOM_COOK_6_THRESHOLD, SQR_TOOM_COOK_85_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, ScratchBuffer,
    TOOM_COOK_4_THRESHOLD, TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD,
    TOOM8_FULL_GUARD_PRODUCT_MIN_SPLIT_LIMBS, TOOM85_PAIRED_RECONSTRUCTION_MIN_LIMBS,
    TRANSFORM_MAX_OPERAND_RATIO, TRANSFORM_MIN_SMALLER_LIMBS,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{
    SQR_SSA_THRESHOLD, SSA_BASE_MODULUS_BITS, SSA_BASECASE_COST_WEIGHT_16THS,
    SSA_BNM1_BASECASE_LIMBS, SSA_COEFFICIENT_VISIT_OVERHEAD, SSA_DIRECT_SHIFT_MAX_LIMBS,
    SSA_NEGACYCLIC_FACTOR3_THRESHOLD, SSA_NEGACYCLIC_FACTOR5_THRESHOLD,
    SSA_NESTED_COST_PENALTY_16THS, SSA_THRESHOLD,
};

use recursive::Recursive;
use shared::{AddMulKernel, SharedEval};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
use ssa::{FftPlan, SsaCrt, SsaRing, SsaTransform};

mod basecase;
mod dispatch;
mod entry;
mod karatsuba;
mod lopsided;
mod low;
mod ntt;
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

#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
mod transform_bench;

pub use basecase::Schoolbook;
pub use dispatch::{MulPlan, MulShape, Multiplication, SquarePlan, TierCeiling, Widths};
pub use entry::MulScratch;
pub use karatsuba::Karatsuba;
pub use lopsided::Lopsided;
pub use low::LowProduct;
pub use ntt::Ntt;
#[cfg(feature = "_internal-tune")]
pub use ntt::{NttMultiplicationPlan, TransformPlan};
#[cfg(not(target_pointer_width = "16"))]
pub use ssa::{Ssa, TransformChoice};
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use ssa::{SsaMultiplicationPlan, SsaSquaringPlan};
pub use toom3::Toom3;
pub use toom4::Toom4;
pub use toom6::Toom6;
pub use toom8::Toom8;
pub use toom32::Toom32;
pub use toom43::Toom43;
#[cfg(all(feature = "_internal-tune", not(target_pointer_width = "16")))]
pub use transform_bench::TransformBench;
