//! Newton-Raphson division (reciprocal division).
//!
//! Computes `1/D` via Newton iteration from a basecase seed, doubling the
//! number of valid bits each step using subquadratic multiplication. That gives
//! `O(M(N))` division, beating Burnikel-Ziegler's `O(M(N) log N)` once the
//! divisor is long enough to amortize building the reciprocal.
//!
//! - [`reciprocal`]: builds and refines `V = floor((B^2n - 1) / D)`.
//! - [`divide`]: applies that reciprocal, block by block, and corrects the
//!   estimate into an exact quotient and remainder.
//!
//! Both contribute their entry points to [`Division`](super::Division);
//! everything else here is private to the file that defines it.

#![allow(
    unsafe_code,
    reason = "Low-level division uses raw pointer access on limb slices to bypass bounds checks, ensuring branchless arithmetic paths in the division kernels."
)]

mod divide;
mod reciprocal;

// The Newton submodules reach the rest of the division tree only through the
// bindings below, which keeps every relative path here one level deep.
use super::{
    Addition, ArchKernels, DivScratch, Division, InternalArbiUint, Limb, LowProduct,
    Multiplication, NEWTON_RAPHSON_BASECASE_LIMBS, ScratchBuffer,
};
