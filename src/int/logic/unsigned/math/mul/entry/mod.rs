//! Multiplication and squaring entry points.
//!
//! Owned storage handles allocation, assignment, and in-place arithmetic.
//! Limb entries select plans and prepare reusable scratch. Workspace drivers
//! execute owned products using bounded stack frames or pooled storage.
//!
//! The owned `a * b` path establishes the complete execution contract:
//! normalized nonzero operands, an exact `a.len() + b.len()` destination, and
//! scratch derived from the selected plan. Valid Rust slice widths prove that
//! sum cannot overflow on any supported limb width. SSA performs its remaining
//! fallible geometry arithmetic while building an operand-bound plan; execution
//! below that boundary contains only diagnostic assertions for proved invariants.

use super::{
    ArchKernels, INLINE_LIMBS, InternalMpUint, KARATSUBA_THRESHOLD, Karatsuba, Limb, LimbOutput,
    MulPlan, Multiplication, SQR_KARATSUBA_THRESHOLD, Schoolbook, ScratchBuffer, SquarePlan,
    TierCeiling, UintRepr,
};
#[cfg(not(target_pointer_width = "16"))]
use super::{LargePlan, Ssa, SsaMultiplicationPlan, TransformChoice};

mod limbs;
mod owned;
mod workspace;

pub use limbs::MulScratch;
