//! Finite-width bitwise operations, bit access, scans, and shifts.

use super::{
    ArchKernels, BoundedPrecision, INLINE_LIMBS, InternalMpUint, LIMB_BITS, LIMB_BYTES, Limb,
    UintRepr,
};

mod access;
mod binary;
mod scan;
mod shift;

#[cfg(test)]
mod tests;
