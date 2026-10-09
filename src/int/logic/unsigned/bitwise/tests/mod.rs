//! Bit access, logical operations, scans, and shifts.

use super::{INLINE_LIMBS, InternalMpUint, LIMB_BITS, Limb};

mod access;
mod binary;
mod scan;
mod shift;
