//! Division for internal big integers.
//!
//! Two names leave this folder. [`Division`] is the division machinery itself —
//! every file below contributes its `impl Division` block, so the tower reads as one
//! surface despite being spread across the folder. `InternalMpUint` gains only
//! the five operations a caller cannot do without (`div_rem`, `div`, `rem`, and
//! the two in-place forms), plus `extended_gcd` and `mod_inverse`; importing the
//! type is enough to use them.
//!
//! The tower, from the bottom up:
//!
//! - [`limbs`]: single-limb division and the in-place add/sub primitives every
//!   kernel shares.
//! - [`reciprocal3by2`]: the Möller-Granlund 3-by-2 division primitive.
//! - [`knuth`]: Knuth's Algorithm D, the basecase all recursion terminates in.
//! - [`burnikel`]: Burnikel-Ziegler recursive division for the middle band.
//! - [`newton`]: Newton-Raphson reciprocal division for the largest divisors.
//! - [`quotient`]: the truncated quotient-only path, which skips the remainder
//!   entirely when the quotient is far shorter than the divisor.
//! - [`bezout`]: extended Euclid and modular inversion, which drive the tower
//!   rather than extend it.
//!
//! [`dispatch`] owns the entry points and picks between those kernels;
//! [`scratch`] owns the reusable buffers they all write into.

#![allow(
    unsafe_code,
    reason = "Low-level division uses raw pointer access on limb slices to bypass bounds checks, ensuring branchless arithmetic paths in the division kernels."
)]

mod bezout;
mod burnikel;
mod dispatch;
mod knuth;
mod limbs;
mod newton;
mod quotient;
mod reciprocal3by2;
mod scratch;

#[cfg(test)]
mod tests;

pub use self::{dispatch::Division, scratch::DivScratch};
// Only the property tests recombine a quotient-remainder pair, so importing
// these unconditionally would be an unused import in normal builds.
// The division submodules reach the surrounding `math` module only through the
// bindings below, which keeps every relative path in this tree one level deep.
use super::{
    Addition, ArchKernels, BURNIKEL_ZIEGLER_BLOCK_LIMBS, BURNIKEL_ZIEGLER_THRESHOLD, DoubleLimb,
    Gcd, InternalMpUint, LIMB_BITS, Limb, LowProduct, MulScratch, Multiplication,
    NEWTON_RAPHSON_BASECASE_LIMBS, NEWTON_RAPHSON_THRESHOLD, ScratchBuffer,
};
