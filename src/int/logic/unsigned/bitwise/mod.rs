//! Bitwise operation implementations for `InternalMpUint`.
//!
//! This module provides bitwise logical operations (AND, OR, XOR, NOT),
//! bit scanning (count_ones, trailing_zeros, find_first_set_bit, etc.),
//! and shift operations (left/right shift, rotate).

use super::{ArchKernels, INLINE_LIMBS, InternalMpUint, LIMB_BITS, LIMB_BYTES, Limb, UintRepr};

mod access;
mod binary;
mod scan;
mod shift;
