//! Division for internal big integers.
//!
//! [`Division`] groups arithmetic kernels, [`DivScratch`] owns reusable storage,
//! and [`bezout::BezoutResult`] represents extended-GCD coefficients. `InternalMpUint`
//! supplies value and assignment operations, division rounding and divisibility.
//!
//! The tower, from the bottom up:
//!
//! - [`single`]: single-limb division and its 2/1 reciprocal primitive.
//! - [`limbs`]: modular residues, normalization, and shared carry propagation.
//! - [`reciprocal3by2`]: the Möller-Granlund 3-by-2 division primitive.
//! - [`knuth`]: Knuth's Algorithm D, the basecase all recursion terminates in.
//! - [`prepared`]: invariant reciprocal and kernel shared by normalized windows.
//! - [`burnikel`]: Burnikel-Ziegler recursive division for the middle band.
//! - [`approximate`]: recursive upper quotient bounds without final residues.
//! - [`newton`]: block-wise Barrett division with a Newton reciprocal whose
//!   precision follows the quotient width.
//! - [`quotient`]: quotient estimates from leading operand prefixes, certified
//!   by retained remainders or bounded corrections.
//! - [`short`]: quotient-only basecase with triangular remainder updates.
//! - [`extended`]: extended Euclid with retained HGCD matrices and cofactor batching.
//! - [`bezout`]: modular inversion and extended-GCD coefficient reconstruction.
//!
//! [`dispatch`] owns the entry points and picks between those kernels;
//! [`scratch`] owns the reusable buffers they all write into.

use super::{
    APPROXIMATE_DIVISION_BLOCK_LIMBS, Addition, ArchKernels, BURNIKEL_LONG_QUOTIENT_THRESHOLD,
    BURNIKEL_QUOTIENT_THRESHOLD, BURNIKEL_ZIEGLER_BLOCK_LIMBS, BURNIKEL_ZIEGLER_THRESHOLD,
    DIVISION_BASECASE_QUOTIENT_MAX_LIMBS, DIVISION_DIVISIBLE_THRESHOLD,
    DIVISION_SINGLE_NORMALIZED_PREINVERSE, DIVISION_SINGLE_UNNORMALIZED_PREINVERSE,
    DIVISION_SMALL_QUOTIENT_MAX, DIVISION_STACK_LIMBS, DIVISION_TRUNCATION_RATIO, DoubleLimb,
    EXTENDED_GCD_COFACTOR_BATCH_MIN_LIMBS, EXTENDED_GCD_COFACTOR_BATCH_RATIO,
    EXTENDED_GCD_WIDE_THRESHOLD, EXTENDED_HGCD_CROSSOVER_THRESHOLD, Gcd, HgcdMatrix, HgcdWorkspace,
    HighProduct, InternalMpUint, LIMB_BITS, Limb, LowProduct, MulScratch, Multiplication,
    NEWTON_QUOTIENT_THRESHOLD, NEWTON_RAPHSON_BASECASE_LIMBS, NEWTON_RAPHSON_THRESHOLD,
    NEWTON_SMALL_QUOTIENT_BLOCK_RATIO, ScratchBuffer,
};

mod approximate;
mod bezout;
mod burnikel;
mod dispatch;
mod divisibility;
mod extended;
mod guarded;
mod knuth;
mod limbs;
mod newton;
mod normalized;
mod prepared;
mod quotient;
mod reciprocal3by2;
mod scratch;
mod short;
mod single;
mod small;

pub use dispatch::Division;
pub use prepared::PreparedDivisor;
pub use scratch::DivScratch;

#[cfg(test)]
mod tests;
