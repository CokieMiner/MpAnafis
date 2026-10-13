//! Scalar division, normalized reciprocal windows, and Knuth's Algorithm D.

use super::{
    Addition, ArchKernels, CompleteDivisionMode, DIVISION_SINGLE_NORMALIZED_PREINVERSE,
    DIVISION_SINGLE_UNNORMALIZED_PREINVERSE, DIVISION_SMALL_QUOTIENT_MAX, DIVISION_STACK_LIMBS,
    DivScratch, Division, DivisionMode, DoubleLimb, InternalMpUint, LIMB_BITS, Limb,
    LimbWriteGuard, QuotientAndRemainder, QuotientDivisionMode, ScratchBuffer,
};

mod fixed;
mod four;
mod knuth;
mod limbs;
mod prepared;
mod reciprocal3by2;
mod single;
mod small;
mod three;

pub use prepared::PreparedDivisor;

#[cfg(test)]
mod tests;
