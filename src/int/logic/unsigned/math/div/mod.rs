//! Division for internal big integers.
//!
//! [`entry`] validates operand geometry and selects value, assignment, exact,
//! divisibility, and reusable-output paths. [`basecase`] owns scalar division,
//! reciprocals, normalization, and Algorithm D. [`recursive`] owns
//! Burnikel-Ziegler, guarded quotient estimates, and Newton division.
//! [`extended`] reconstructs extended-GCD coefficients and modular inverses.
//!
//! [`Division`] groups shared kernels, [`DivScratch`] owns reusable storage,
//! and [`output`] specifies compile-time output modes and bounded certificates.

use super::{
    APPROXIMATE_DIVISION_BLOCK_LIMBS, Addition, ArchKernels, BURNIKEL_LONG_QUOTIENT_THRESHOLD,
    BURNIKEL_QUOTIENT_THRESHOLD, BURNIKEL_ZIEGLER_BLOCK_LIMBS, BURNIKEL_ZIEGLER_THRESHOLD,
    DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, DIVISION_DIVISIBLE_THRESHOLD,
    DIVISION_SINGLE_NORMALIZED_PREINVERSE, DIVISION_SINGLE_UNNORMALIZED_PREINVERSE,
    DIVISION_SMALL_QUOTIENT_MAX, DIVISION_STACK_LIMBS, DIVISION_TRUNCATION_RATIO, DoubleLimb,
    EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS, EXTENDED_GCD_COFACTOR_BATCH_RATIO,
    EXTENDED_GCD_WIDE_THRESHOLD, EXTENDED_HGCD_CROSSOVER_THRESHOLD, Gcd, HgcdMatrix, HgcdWorkspace,
    HighProduct, InternalMpUint, LIMB_BITS, Limb, LimbWriteGuard, LowProduct, MulScratch,
    Multiplication, NEWTON_QUOTIENT_THRESHOLD, NEWTON_RAPHSON_BASECASE_LIMBS,
    NEWTON_RAPHSON_THRESHOLD, NEWTON_SMALL_QUOTIENT_BLOCK_RATIO, ScratchBuffer,
};

mod basecase;
mod entry;
mod extended;
mod output;
mod recursive;
mod scratch;

pub use basecase::PreparedDivisor;
pub use entry::Division;
pub use output::{
    CompleteDivisionMode, DivisionAttempt, DivisionMode, QuotientAccuracy, QuotientAndRemainder,
    QuotientDivisionMode, QuotientOnly, RemainderCertificate, RemainderOnly, TruncatedPrefix,
};
pub use scratch::DivScratch;
