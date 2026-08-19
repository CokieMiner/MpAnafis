//! Signed integer implementation — `InternalMpInt`, arithmetic, bitwise, comparison.

use super::{INLINE_LIMBS, InternalMpUint, Limb, UintRepr};

use assignment::negate_normalized_inplace;

mod arithmetic;
mod assignment;
mod bitwise;
mod cmp;
mod division;
mod mpint;
mod theory;
mod wrapping;

pub use mpint::InternalMpInt;
