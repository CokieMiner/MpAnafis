//! Addition and subtraction for internal big integers.
//!
//! For limb base `B = 2^LIMB_BITS`, each column adds or subtracts at most two
//! limbs and a binary carry or borrow. Limb arithmetic therefore computes a
//! residue modulo B and propagates a flag in {0, 1}; its overflow behavior is
//! part of the mathematical operation. Lengths and indices do not wrap:
//! unchecked arithmetic requires the local order or representation bound.
//!
//! A normalized sum needs no high-zero scan. Subtraction removes zero high
//! limbs from its residue and retains the borrow when underflow is requested.
//!
//! `sub` and `sub_assign` require a nonnegative difference. Bounded subtraction
//! sign-extends a negative residue before masking a partial destination limb.
//!
//! References:
//! - D. E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*,
//!   3rd ed., Addison-Wesley, 1997, Section 4.3.1, Algorithm A & Algorithm S.
//! - R. P. Brent and P. Zimmermann, *Modern Computer Arithmetic*, Cambridge University
//!   Press, 2011, Section 1.2. DOI: 10.1017/CBO9780511921698.

use super::{ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb, UintRepr};

mod assign;
mod fused;
mod limbs;
mod step;
mod values;

pub use limbs::Addition;

#[cfg(test)]
mod tests;
