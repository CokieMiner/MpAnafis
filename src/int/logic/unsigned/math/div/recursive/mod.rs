//! Recursive division, upper quotient bounds, and guarded correction.

use super::{
    APPROXIMATE_DIVISION_BLOCK_LIMBS, Addition, ArchKernels, BURNIKEL_QUOTIENT_THRESHOLD,
    BURNIKEL_ZIEGLER_BLOCK_LIMBS, CompleteDivisionMode, DivScratch, Division, HighProduct,
    InternalMpUint, LIMB_BITS, Limb, LowProduct, MulScratch, Multiplication,
    NEWTON_QUOTIENT_THRESHOLD, NEWTON_RAPHSON_BASECASE_LIMBS, NEWTON_SMALL_QUOTIENT_BLOCK_RATIO,
    PreparedDivisor, QuotientAccuracy, QuotientOnly, RemainderCertificate, RemainderOnly,
    ScratchBuffer, TruncatedPrefix,
};

mod approximate;
mod burnikel;
mod guarded;
mod newton;

#[cfg(test)]
mod tests;
