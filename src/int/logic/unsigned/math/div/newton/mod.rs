//! Block-wise Barrett division with Newton reciprocal construction.
//!
//! Newton refinement doubles reciprocal precision using subquadratic
//! multiplication. Balanced division requires `O(M(n))` work; longer
//! dividends reuse the reciprocal across quotient blocks.
//!
//! - [`reciprocal`]: builds a lower approximation to `B^(n+k)/D`, with
//!   guard limbs bounding its absolute error below two.
//! - [`basecase`]: constructs the exact seed from a complemented numerator.
//! - [`divide`]: applies that reciprocal, block by block, and corrects the
//!   estimate into an exact quotient and remainder.
//! - [`estimate`]: forms lower quotient bounds and exact-division corrections.
//! - [`products`]: reconstructs residues and forms reciprocal corrections.
//!
//! High products and their carry certificates are owned by [`HighProduct`].
//!
//! [`Division`] groups reciprocal construction and block evaluation.
//!
//! References:
//! - A. Schönhage, "Asymptotically fast algorithms for GCD of polynomials and division of polynomials and integers", Proc. ISSAC 1982, pp. 182–183.
//! - R. P. Brent and P. Zimmermann, *Modern Computer Arithmetic*, Cambridge University
//!   Press, 2011, Section 1.4 "Division". DOI: 10.1017/CBO9780511921698.
//! - GNU MP, [Block-Wise Barrett Division](https://gmplib.org/manual/Block_002dWise-Barrett-Division).

#[cfg(not(target_pointer_width = "16"))]
use super::Multiplication;
use super::{
    Addition, ArchKernels, DivScratch, Division, HighProduct, InternalMpUint, Limb, LowProduct,
    MulScratch, NEWTON_RAPHSON_BASECASE_LIMBS, NEWTON_SMALL_QUOTIENT_BLOCK_RATIO, PreparedDivisor,
    ScratchBuffer,
};

mod basecase;
mod divide;
mod estimate;
mod products;
mod reciprocal;

#[cfg(test)]
mod tests;
