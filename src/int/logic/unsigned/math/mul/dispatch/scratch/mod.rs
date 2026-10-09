//! Plan workspace sizing and conventional tier recurrences.

use super::{
    KARATSUBA_THRESHOLD, Karatsuba, Lopsided, MulPlan, MulShape, Multiplication,
    SQR_KARATSUBA_THRESHOLD, SQR_TOOM_COOK_THRESHOLD, SquarePlan, TOOM_COOK_THRESHOLD, TierCeiling,
    Toom6, Toom8, Widths,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, Ssa};

mod plan;
mod tiers;
