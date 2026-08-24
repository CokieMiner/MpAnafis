//! Fermat ring arithmetic on fixed-width limb slices.
//!
//! All operations work in $\mathbb{Z}/(2^n + 1)$ where `n = mod_bits` is a
//! multiple of `LIMB_BITS`. Coefficients are stored in fixed-width slots of
//! `coeff_limbs = mod_bits / LIMB_BITS + 1` limbs, where the guard limb at
//! position `mod_limbs` is 0 or 1. Transform butterflies use a semi-normalized
//! representation with arbitrary data limbs and a guard at most one; point
//! products and reconstruction canonicalize to `[0, 2^n]` at their boundaries.
//!
//! The ring width is *not* required to be a power of two: the planner emits
//! alignment-derived widths such as 4224 or 12288 bits, and every routine here
//! is written for the general case.
//!
//! Every file here contributes to the single [`SsaRing`] namespace:
//! - [`arith`]: declares it, plus slot widths, shift-period reduction, and the
//!   add/subtract/negate/normalize family with its carry propagation.
//! - [`shift`]: in-place multiplication by a power of two, including the
//!   half-bit steps that carry a `sqrt(2)` factor.
//! - [`shift_from`]: out-of-place multiplication by a power of two.

#![allow(
    unsafe_code,
    reason = "Fermat-ring slice arithmetic uses raw pointer kernels for peak FFT performance"
)]

use super::{Addition, ArchKernels, LIMB_BITS, Limb, SSA_DIRECT_SHIFT_MAX_LIMBS, SsaCarry};

mod arith;
mod shift;
mod shift_from;

pub use arith::{Residue, SsaRing};
