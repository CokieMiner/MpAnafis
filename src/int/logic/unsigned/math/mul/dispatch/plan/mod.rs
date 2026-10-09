//! Multiplication-tier plan types, operand-shape policy, selection, and execution.

use super::{
    BALANCED_TOOM8_THRESHOLD, KARATSUBA_THRESHOLD, Karatsuba, Limb, LimbOutput, Lopsided,
    SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_4_THRESHOLD, SQR_TOOM_COOK_6_THRESHOLD,
    SQR_TOOM_COOK_85_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, Schoolbook, TOOM_COOK_4_THRESHOLD,
    TOOM_COOK_6_THRESHOLD, TOOM_COOK_85_THRESHOLD, TOOM_COOK_THRESHOLD, Toom3, Toom4, Toom6, Toom8,
    Toom32, Toom43,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{
    SQR_SSA_THRESHOLD, SSA_THRESHOLD, Ssa, SsaPlan, TRANSFORM_MAX_OPERAND_RATIO,
    TRANSFORM_MIN_SMALLER_LIMBS, TransformChoice,
};

mod execute;
mod select;
mod shape;
mod types;

#[cfg(not(target_pointer_width = "16"))]
pub use types::LargePlan;
pub use types::{MulPlan, MulShape, Multiplication, SquarePlan, TierCeiling, Widths};
