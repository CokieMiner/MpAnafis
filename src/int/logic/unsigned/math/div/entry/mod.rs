//! Value operations, assignment, reusable outputs, and division dispatch.

use super::{
    ArchKernels, BURNIKEL_LONG_QUOTIENT_THRESHOLD, BURNIKEL_QUOTIENT_THRESHOLD,
    BURNIKEL_ZIEGLER_THRESHOLD, CompleteDivisionMode, DIVISION_BASECASE_QUOTIENT_MAX_LIMBS,
    DIVISION_DIVISIBLE_THRESHOLD, DIVISION_STACK_LIMBS, DIVISION_TRUNCATION_RATIO, DivScratch,
    DivisionAttempt, DoubleLimb, InternalMpUint, LIMB_BITS, Limb, Multiplication,
    NEWTON_QUOTIENT_THRESHOLD, NEWTON_RAPHSON_THRESHOLD, QuotientAndRemainder, QuotientOnly,
    RemainderCertificate, RemainderOnly, ScratchBuffer, TruncatedPrefix,
};

mod dispatch;
mod divisibility;
mod exact;
mod normalized;
mod quotient;
mod value;

pub use dispatch::Division;

#[cfg(test)]
mod tests;
